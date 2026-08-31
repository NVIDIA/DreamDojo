# Latent Action Model Training

> [!TIP]
> For accepted-only UMI continued pretraining, multi-node checkpointing,
> action-centered evaluation, and the complete 500/2k/10k reference results,
> see the [LAM continuation report](../external/lam_project/README.md).

The provided configuration only includes the GR-1 post-training dataset as an example. You can add new datasets by adding folders to `dataset_paths` by yourselves. The code will recursively search the folder and add any detected MP4 videos to the training set.

Using the following bash script to launch latent action training (suppose you are training on 1 node with 8 GPUs).

```
cd external/lam_project
bash train.sh
```

A multi-node training script example is provided at `external/lam_project/launch.sh`. Add the index of the YAML file to specify the configuration to use.

```
bash launch.sh 0
```

> [!NOTE]
> Our latent action model weights can be found at [Hugging Face](https://huggingface.co/nvidia/DreamDojo). To set the path to your own checkpoint for proxy action extraction, check `https://github.com/NVIDIA/DreamDojo/blob/main/cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py`.

---

<= Previous: [[Setup](https://github.com/NVIDIA/DreamDojo/blob/main/docs/SETUP.md)]

=> Next: [[DreamDojo Pretraining](https://github.com/NVIDIA/DreamDojo/blob/main/docs/PRETRAIN.md)]
