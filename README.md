# EEG Challenge 2025 Hall of Fame

This repository contains the top submissions to the EEG Foundation Challenge 2025, showcasing state-of-the-art approaches for EEG signal analysis and processing.

## 📊 About the Challenge

The EEG Foundation Challenge 2025 focuses on advancing electroencephalography (EEG) signal analysis through innovative machine learning and deep learning approaches. The challenge features multiple tasks designed to push the boundaries of EEG signal understanding and clinical applications.

### Challenge Tasks

- **Challenge 1**: Primary EEG analysis task
- **Challenge 2**: Advanced EEG processing task

## 🏆 Leaderboard

The following teams have achieved top rankings in the EEG Foundation Challenge 2025:

| Team | Models | Weights | Challenge 1 rank | Challenge 2 rank |
|------|------------|------------|------------------|------------------|
| KU_Leuven | [GitHub](KU_Leuven/) | [HuggingFace](https://huggingface.co/eeg2025/KU_Leuven) | 1 | / |
| Sigma_Nova | [GitHub](Sigma_Nova/) | [HuggingFace](https://huggingface.co/eeg2025/Sigma-Nova) | 2 | / |
| MBZUAI | [GitHub](MBZUAI/) | [HuggingFace](https://huggingface.co/eeg2025/MBZUAI) | 5 | 2 |
| MIND_CICO | [GitHub](MIND_CICO/) | [HuggingFace](https://huggingface.co/eeg2025/MIND-CICO) | 3 | 3 |

**Note**: "/" indicates the team did not participate in that specific challenge.

## 🚀 Getting Started

### Requirements

- Python 3.8+
- PyTorch
- Additional dependencies are specified in each team's submission files

### Installation

Clone this repository:

```bash
git clone https://github.com/eeg2025/hall-of-fame-2025.git
cd hall-of-fame-2025
```

### Usage Examples

Each team's submission can be run independently. The models will automatically download their weights from HuggingFace when executed.

#### KU_Leuven (Rank 1 - Challenge 1)

```bash
python ./KU_Leuven/submission.py
```

#### Sigma_Nova (Rank 2 - Challenge 1)

```bash
python ./Sigma_Nova/submission.py
```

#### MBZUAI (Rank 5 - Challenge 1, Rank 2 - Challenge 2)

```bash
python ./MBZUAI/submission.py
```

#### MIND_CICO (Rank 3 - Challenge 1, Rank 3 - Challenge 2)

```bash
python ./MIND_CICO/submission.py
```

## 📁 Repository Structure

```
hall-of-fame-2025/
├── KU_Leuven/           # Top submission for Challenge 1
├── Sigma_Nova/          # Second place for Challenge 1
├── MBZUAI/              # Top-5 Challenge 1, Runner-up Challenge 2
├── MIND_CICO/           # Top submissions for both challenges
├── LICENSE              # BSD 3-Clause License
└── README.md            # This file
```

Each team directory contains:
- `submission.py` - Main submission script with model definitions and inference code
- Supporting files and model architectures as needed

## 📝 Submission Format

All submissions follow a standardized format:
- Models are defined within the submission scripts
- Pre-trained weights are hosted on HuggingFace and automatically downloaded
- Each submission is self-contained and can run independently

## 🙏 Acknowledgements

We thank all participants of the EEG Foundation Challenge 2025 for their innovative contributions to advancing EEG signal analysis. Special recognition goes to:

- **KU_Leuven** - For achieving first place in Challenge 1
- **Sigma_Nova** - For their outstanding performance in Challenge 1
- **MBZUAI** - For excellent results across both challenges
- **MIND_CICO** - For consistent top-3 performance in both challenges

## 📄 License

This repository is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

## 🔗 Links

- Challenge Website: [EEG Foundation Challenge 2025](https://github.com/eeg2025)
- Model Weights: [HuggingFace - eeg2025](https://huggingface.co/eeg2025)

## 📧 Contact

For questions about the challenge or submissions, please refer to the official EEG Challenge 2025 channels.
