## ROAM
This is the **PyTorch implementation** of [**Beyond Routines: Adaptive Mobility Prediction via Sequential-Relational Fusion**](https://doi.org/10.1145/3770854.3780268).

### Datasets

For access to **raw datasets**, please refer to the [Humob Challenge 2024](https://wp.nyu.edu/humobchallenge2024/). Follow the instructions on their website for data access and usage policies.

We processed the raw dataset by removing consecutive duplicate records in order to extract meaningful user activity locations. The processed data were then split into `train.csv`, `val.csv`, and `test.csv` files. All files contain the following five attributes, consistent with the original dataset: `uid`, `d`, `t`, `x`, `y`.  

### Running Steps
* Optional arguments can be modified in train.py.
* To train the model, run:
   ```bash
   python train.py
* To evaluate the trained model, run:
   ```bash
   python train.py --test

### Citation
```bibtex
@inproceedings{sun2026beyond,
  title={Beyond Routines: Adaptive Mobility Prediction via Sequential-Relational Fusion},
  author={Sun, Tianao and Liu, Ruizhe and Jia, Wenzhen and Zhao, Kai and Huang, Weiming and Chen, Meng},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1},
  year={2026}
}
