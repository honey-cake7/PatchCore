import contextlib
import gc
import logging
import os
import sys

import click
import numpy as np
import torch

import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.ttt4as
import patchcore.utils

LOGGER = logging.getLogger(__name__)

_DATASETS = {
    "mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"],
    "kvasir": ["patchcore.datasets.kvasir", "KvasirDataset"]
}


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--ttt4as", is_flag=True, help="Enable TTT4AS binary segmentation.")
@click.option(
    "--ttt4as_features",
    type=click.Choice(["wrn50", "dinov2"]),
    default="wrn50",
    show_default=True,
    help="Feature extractor used to train the test-time SVM.",
)
@click.option(
    "--percentile",
    type=float,
    default=99.0,
    show_default=True,
    help="Percentile for TTT4AS peak suppression / THR fallback.",
)
@click.option(
    "--thr_sigma",
    type=float,
    default=3.0,
    show_default=True,
    help="c in the mu + c*sigma threshold baseline.",
)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    save_segmentation_images,
    ttt4as,
    ttt4as_features,
    percentile,
    thr_sigma,
):
    methods = {key: item for (key, item) in methods}

    os.makedirs(results_path, exist_ok=True)

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    dataloader_iter, n_dataloaders = methods["get_dataloaders_iter"]
    dataloader_iter = dataloader_iter(seed)
    patchcore_iter, n_patchcores = methods["get_patchcore_iter"]
    patchcore_iter = patchcore_iter(device)
    if not (n_dataloaders == n_patchcores or n_patchcores == 1):
        raise ValueError(
            "Please ensure that #PatchCores == #Datasets or #PatchCores == 1!"
        )

    for dataloader_count, dataloaders in enumerate(dataloader_iter):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["testing"].name, dataloader_count + 1, n_dataloaders
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["testing"].name

        with device_context:

            torch.cuda.empty_cache()
            if dataloader_count < n_patchcores:
                PatchCore_list = next(patchcore_iter)

            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
                    dataloaders["testing"]
                )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # Plot Example Images.
            if save_segmentation_images:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                def image_transform(image):
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                patchcore.utils.plot_segmentation_images(
                    results_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            LOGGER.info("Computing evaluation metrics.")
            # Compute Image-level AUROC scores for all images.
            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )["auroc"]

            # Compute PRO score & PW Auroc for all images
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            full_pixel_auroc = pixel_scores["auroc"]

            # Compute PRO score & PW Auroc only for images with anomalies
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs], [masks_gt[i] for i in sel_idxs]
            )
            anomaly_pixel_auroc = pixel_scores["auroc"]

            result_dict = {
                "dataset_name": dataset_name,
                "instance_auroc": auroc,
                "full_pixel_auroc": full_pixel_auroc,
                "anomaly_pixel_auroc": anomaly_pixel_auroc,
            }

            # TTT4AS: binary segmentation via per-image test-time SVM, compared
            # against the mu + c*sigma threshold baseline, on anomalous samples.
            if ttt4as and sel_idxs:
                if len(PatchCore_list) > 1:
                    LOGGER.warning(
                        "TTT4AS uses only the first PatchCore of the ensemble."
                    )
                ad_model = PatchCore_list[0]
                # Raw (un-normalized) score maps from the AD&S model.
                raw_test_segs = aggregator["segmentations"][0]
                gt_anom = [masks_gt[i] for i in sel_idxs]

                LOGGER.info("Fitting THR baseline from validation scores...")
                _, val_segs, _, _ = ad_model.predict(dataloaders["validation"])
                val_segs = np.stack(val_segs)
                thr_map = val_segs.mean(axis=0) + thr_sigma * val_segs.std(axis=0)
                thr_bin = [
                    (np.asarray(raw_test_segs[i]) > thr_map).astype(np.uint8)
                    for i in sel_idxs
                ]
                thr_metrics = patchcore.metrics.compute_binary_segmentation_metrics(
                    thr_bin, gt_anom
                )

                LOGGER.info(
                    "Running TTT4AS ({} features) on {} anomalous samples...".format(
                        ttt4as_features, len(sel_idxs)
                    )
                )
                ttt = patchcore.ttt4as.TTT4AS(
                    ad_model,
                    feature_extractor=ttt4as_features,
                    device=device,
                    percentile=percentile,
                    seed=seed,
                )
                ttt_bin = []
                for idx in sel_idxs:
                    image = dataloaders["testing"].dataset[idx]["image"]
                    feature_map = ttt.extract_feature_map(image)
                    ttt_bin.append(
                        ttt.predict_binary_map(
                            raw_test_segs[idx], feature_map, image_index=idx
                        )
                    )
                ttt_metrics = patchcore.metrics.compute_binary_segmentation_metrics(
                    ttt_bin, gt_anom
                )

                result_dict.update(
                    {
                        "thr_precision": thr_metrics["precision"],
                        "thr_recall": thr_metrics["recall"],
                        "thr_f1": thr_metrics["f1"],
                        "ttt4as_precision": ttt_metrics["precision"],
                        "ttt4as_recall": ttt_metrics["recall"],
                        "ttt4as_f1": ttt_metrics["f1"],
                    }
                )

            result_collect.append(result_dict)

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            del PatchCore_list
            gc.collect()

        LOGGER.info("\n\n-----\n")

    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        results_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


@main.command("patch_core_loader")
# Pretraining-specific parameters.
@click.option("--patch_core_paths", "-p", type=str, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core_loader(patch_core_paths, faiss_on_gpu, faiss_num_workers):
    def get_patchcore_iter(device):
        for patch_core_path in patch_core_paths:
            loaded_patchcores = []
            gc.collect()
            n_patchcores = len(
                [x for x in os.listdir(patch_core_path) if ".faiss" in x]
            )
            if n_patchcores == 1:
                nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
                patchcore_instance = patchcore.patchcore.PatchCore(device)
                patchcore_instance.load_from_path(
                    load_path=patch_core_path, device=device, nn_method=nn_method
                )
                loaded_patchcores.append(patchcore_instance)
            else:
                for i in range(n_patchcores):
                    nn_method = patchcore.common.FaissNN(
                        faiss_on_gpu, faiss_num_workers
                    )
                    patchcore_instance = patchcore.patchcore.PatchCore(device)
                    patchcore_instance.load_from_path(
                        load_path=patch_core_path,
                        device=device,
                        nn_method=nn_method,
                        prepend="Ensemble-{}-{}_".format(i + 1, n_patchcores),
                    )
                    loaded_patchcores.append(patchcore_instance)

            yield loaded_patchcores

    return ("get_patchcore_iter", [get_patchcore_iter, len(patch_core_paths)])


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=1, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option(
    "--train_val_split",
    default=0.8,
    type=float,
    show_default=True,
    help="Fraction of train kept as train; the rest forms the nominal val set "
    "used by the TTT4AS THR baseline.",
)
@click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    subdatasets,
    batch_size,
    resize,
    imagesize,
    num_workers,
    train_val_split,
    augment,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders_iter(seed):
        for subdataset in subdatasets:
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader.name = name
            if subdataset is not None:
                test_dataloader.name += "_" + subdataset

            # Nominal-only validation split carved out of the train folder,
            # used to fit the mu + c*sigma threshold baseline for TTT4AS.
            val_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.VAL,
                train_val_split=train_val_split,
                seed=seed,
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            dataloader_dict = {
                "testing": test_dataloader,
                "validation": val_dataloader,
            }

            yield dataloader_dict

    return ("get_dataloaders_iter", [get_dataloaders_iter, len(subdatasets)])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
