# CreativeGen Book

영문 한 문장을 받아 왼쪽에는 삽화, 오른쪽에는 원문이 인쇄된 책을 만든다. 책 둘레의 배경도 같은 문장에서 생성한다. 배경·삽화·책 모듈은 같은 SDXL Base와 각자 학습한 가중치를 사용한다. 마지막 단계에서 세 이미지를 배치해 역확산하고, 오른쪽 페이지의 글자 마스크에 맞춰 잠재값을 보정한다.

| 코드 | 역할 |
|---|---|
| `architecture/` | SDXL용 LoRA, 글자·페이지 조건 인코더, 모듈별 입력 조건 |
| `pipeline/` | 영문 한 문장에서 배경·삽화·책을 생성하고 최종 이미지를 저장 |
| `training/` | 학습 데이터를 읽고 네 단계를 차례로 학습·검증 |

## 실행

저장소 루트에서 실행한다. 학습 서버 환경은 Linux, Python 3.12.3, PyTorch 2.6.0과 CUDA 12.4이다. 추론에는 20GB GPU 한 장을 사용한다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export SDXL_BASE="$PWD/models/sdxl-base"
```

기존 SDXL Base 사본이 있으면 `SDXL_BASE`에 그 경로를 지정한다. 새로 설치할 때는 아래 명령으로 고정한 리비전에서 필요한 파일만 받는다. 학습과 추론은 실행 전에 각 파일의 SHA-256을 검사한다.

```bash
python - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download

model = json.loads(Path("configs/sdxl-base-files.json").read_text())
snapshot_download(model["model_id"], revision=model["revision"],
                  allow_patterns=list(model["files"]), local_dir=os.environ["SDXL_BASE"])
PY
PYTHON="$(command -v python)" bash training/train.sh
```

| 학습 순서 | 업데이트 | 학습률 | 가중치 저장 위치 |
|---|---:|---:|---|
| 배경 | 240 | 0.00006 | `runs/background/` |
| 삽화 | 240 | 0.00006 | `runs/illustration/` |
| 책 | 128 | 0.00008 | `runs/book/` |
| 글자·페이지 조건을 함께 쓰는 책 | 128 | 0.00008 | `runs/book_joint_01/` |

네 단계 모두 해상도 768, LoRA rank 4를 사용한다. 마지막 단계는 이 실행에서 만든 책 가중치로 초기화한다. 학습 자료와 명세는 `data/`에 있고, 선택된 단계와 검증 손실은 모듈별 `evaluation.json`에 기록한다.

학습이 끝나면 문장과 시드를 넣는다.

```bash
python -m pipeline --model "$SDXL_BASE" \
  --sentence "A seagull flies just above the surface of the water." \
  --seed 241125 --output runs/seagull
```

최종 책은 `runs/seagull/spread/final.png`, 독립 생성한 배경과 삽화는 `background/background.png`, `illustration/illustration.png`에 저장된다. 같은 실행 폴더의 `manifest.json`에 프롬프트·시드·입력 파일 해시가 남는다.

## 자료와 방법

배경은 NGA 공개 사진 60장, 삽화는 British Library의 도판 60장으로 각각 48장을 학습, 12장을 검증에 사용한다. 책은 Internet Archive의 공개 도서 스캔 20건으로 시작해 글자 조건 학습에서 27건을 사용한다. 작품별 링크·권리 표시·분할은 `data/`의 각 명세에 있고, 전사·크롭·쪽 배치의 근거는 `data/sources/`에 있다. 글꼴은 DejaVu Serif이며 이용허락문을 함께 보관한다. [NGA Open Access](https://www.nga.gov/artworks/free-images-and-open-access) · [British Library 도판](https://bl.iro.bl.uk/entities/product/70635ff1-908e-457f-8bbf-f88ce0d9627f) · [책 스캔 출처](data/sources/book_spreads.json) · [추가 스캔](data/sources/book_print_extra.json) · [글꼴 이용허락문](configs/fonts/LICENSE-DejaVu)

모델과 구현에는 SDXL Base, Diffusers, LoRA를 사용했다. 개인화 책 프로젝트의 구성을 초기 아이디어로 참고했다. [SDXL Base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) · [SDXL 논문](https://arxiv.org/abs/2307.01952) · [Diffusers](https://github.com/huggingface/diffusers) · [LoRA](https://arxiv.org/abs/2106.09685) · [참고한 책 프로젝트](https://github.com/yunseopark-kaist/personalized-book-gen)

개발 중 Codex를 코드 작성, 영문 프롬프트 구상, 구도 시안 제작과 샘플 비교에 사용했다. 구도 시안의 활용 및 자동 영역 조합·인쇄 보정에 관한 수업 측 확인은 아직 남아 있다. [CreativeGen 과제 지침](https://generative-models.kaist.ac.kr/creativegen/#important-notes)
