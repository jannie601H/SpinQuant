# GPTQ 프로토콜에 따른 PTQ 실험과 rotation 캐시

프로젝트 루트에서 실행한다. `scripts/` 디렉터리에 있다면 `bash run_ptq.sh ...`를 사용한다.

PyTorch가 설치된 가상환경을 활성화하거나 `PYTHON_BIN`으로 해당 Python 경로를 지정한다. 스크립트는 기본적으로 `python3`를 사용하며, 캐시 처리와 학습·평가 모두 같은 Python으로 실행한다 (`python -m torch.distributed.run`).

```bash
PYTHON_BIN="$HOME/spinquant-env/bin/python" RESPIN=1 bash scripts/run_ptq.sh meta-llama/Llama-3.2-1B 4 8 16 16 gptq on
```

`MODEL`에는 전체 Hugging Face ID (`meta-llama/Llama-3.2-1B`) 또는 `config.json`과 가중치가 있는 로컬 디렉터리 경로를 지정한다. `Llama-3.2-1B`라는 로컬 폴더가 없다면 짧은 모델 이름만으로는 불러올 수 없다.

```bash
bash scripts/run_ptq.sh MODEL W_BITS A_BITS K_BITS V_BITS QUANTIZER ROTATION [HAD]
```

`W_BITS`는 항상 **최종 PTQ의 target weight bit**다. 학습용 weight bit는 스크립트가 별도로 결정한다.

| 최종 PTQ 설정 | Rotation optimization | 최종 PTQ |
|---|---|---|
| GPTQ, rotation on | W16 + target A/K/V | target W/A/K/V, GPTQ |
| RTN, rotation on | target W/A/K/V | target W/A/K/V, RTN |
| GPTQ, rotation off | 실행하지 않음 | target W/A/K/V, GPTQ |
| RTN, rotation off | 실행하지 않음 | target W/A/K/V, RTN |

GPTQ에서는 rotation 학습 중 weight 양자화를 생략하고 activation/KV 양자화 조건을 유지한다. 최종 `ptq.py`가 학습된 rotation을 적용한 가중치를 target bit로 GPTQ 양자화한다. RTN은 기존처럼 target weight bit로 rotation을 학습한다.

## Rotation과 Had 설정

- `RESPIN=1` 환경 변수를 지정하면 전역 R1 대신 층별 A/B를 학습한다. `ROTATION=on`이 필요하며 R2/R3/R4 설정은 동일하게 적용된다.
- `ROTATION`: learned R1/R2 사용 여부 (`on` / `off`).
- `HAD`: R3/R4 Hadamard 적용 여부를 함께 제어하는 8번째 선택 인자 (`on` / `off`). 생략하면 ROTATION을 따른다.
- `HAD=on`: R4는 켜고, R3는 **K<16일 때만** 켠다. K16에서는 K 양자화가 없으므로 R3를 생략한다.
- `HAD=off`: R3/R4 모두 끈다. K4/K8 양자화 자체는 유지한다.
- `ROTATION=off`: No Rotation. HAD도 off여야 하며 R1/R2/R3/R4 모두 끈다. `off on`은 오류다.
- R4는 down_proj 가중치와 입력 activation 변환을 함께 제어한다.

| ROTATION | HAD | K bits | R1/R2 | R3 | R4 |
|---|---|---|---|---|---|
| on | on | 4/8 | on | on | on |
| on | on | 16 | on | off | on |
| on | off | 모두 | on | off | off |
| off | off | 모두 | off | off | off |

스크립트의 이전 `[R3] [R4]` 두 인자는 `[HAD]` 하나로 대체했다. 예전 `on off off` 호출은 `on off`로 바꾼다. 내부 Python에는 계산한 `--r3`/`--no-r3`, `--r4`/`--no-r4`를 전달하며, 직접 Python을 호출하는 저수준 옵션은 그대로 유지한다.

```bash
# 기본 Had 실험: W16A4K16V16 학습 → W4A4K16V16 GPTQ 평가.
bash scripts/run_ptq.sh meta-llama/Llama-3.2-1B 4 4 16 16 gptq on

# Respin: 층별 attention/MLP 좌표계와 residual 변환을 함께 학습·적용.
RESPIN=1 bash scripts/run_ptq.sh meta-llama/Llama-3.2-1B 4 4 16 16 gptq on

# No-Had: R1/R2만 학습·적용.
bash scripts/run_ptq.sh meta-llama/Llama-3.2-1B 4 4 16 16 gptq on off

# Rotation 전체 OFF. 학습 없이 W4A4K4V4 RTN 평가.
bash scripts/run_ptq.sh meta-llama/Llama-3.2-1B 4 4 4 4 rtn off

# 같은 A/K/V·Had·학습 설정의 GPTQ target W8은 W4 실험의 W16 캐시 재사용.
bash scripts/run_ptq.sh meta-llama/Llama-3.2-1B 8 4 16 16 gptq on

# 최종 baseline용 100-step 학습은 10-step 캐시와 별도.
MAX_STEPS=100 bash scripts/run_ptq.sh meta-llama/Llama-3.2-1B 4 4 16 16 gptq on

# 유효한 캐시가 있어도 강제로 다시 학습.
FORCE_ROTATION=1 bash scripts/run_ptq.sh meta-llama/Llama-3.2-1B 4 4 16 16 gptq on
```

## 학습 설정과 저장 구조

| 환경 변수 | 기본값 | 용도 |
|---|---|---|
| `PYTHON_BIN` | `python3` | PyTorch가 설치된 Python 실행 파일 |
| `RESPIN` | `0` | `1`이면 전역 R1 대신 층별 A/B 사용 |
| `MAX_STEPS` | `10` | Rotation 최적화 step 수 |
| `ROTATION_SEED` | `0` | 두 Python 실행에 전달하는 `--seed` |
| `FORCE_ROTATION` | `0` | `1`이면 캐시가 있어도 다시 최적화 |
| `RESULT_DIR` | `results` | PTQ 로그/metadata와 rotation 캐시 저장 루트 |

학습률은 1.5, batch size는 1, sequence length는 2048이며 스크립트의 최적화 인자에서 변경할 수 있다. 기본 10-step은 pipeline 검증용이다. 100-step 실행은 처음부터 학습하며 10-step R.bin에서 resume하지 않는다. R.bin에는 R1/R2 (Respin에서는 A/B/R2)가 있고 optimizer/scheduler 상태가 없다. `max_steps`는 학습량과 cosine 학습률 스케줄에 영향을 주므로 캐시를 구분한다.

Respin은 `A.0`부터 `A.N`까지 N+1개, `B.0`부터 `B.(N-1)`까지 N개의 행렬을 저장한다. PyTorch weight 기준으로 Q/K/V는 `W @ Ai`, o_proj는 `Bi.T @ W`, gate/up은 `W @ Bi`, down은 `A(i+1).T @ W`로 변환한다. 첫 embedding에는 `A0`, 마지막 head에는 `AN`을 fuse한다. PTQ 모델의 두 residual buffer에는 `Ai.T @ Bi`, `Bi.T @ A(i+1)`을 저장한다. 학습 중에는 이 곱을 매 forward에서 계산해 A/B gradient를 유지한다. 저수준 실행에서는 `optimize_rotation.py --respin`, `ptq.py --rotate --respin`을 사용한다. 기존 R1/R2 checkpoint도 계속 지원한다. Respin residual을 지원하지 않는 ExecuTorch export는 오류로 중단한다.

```text
results/
  rotation/
    Llama-3.2-1B_W16A4K16V16_steps10_r3-off_r4-on_<cache-hash>/
      R.bin
      metadata.json
      optimize.log
  Llama-3.2-1B_W4A4K16V16_gptq_rot-on_had-on_r3-off_r4-on_steps10_rotW16_seed-0_<run-hash>.log
  Llama-3.2-1B_W4A4K16V16_gptq_rot-on_had-on_r3-off_r4-on_steps10_rotW16_seed-0_<run-hash>.metadata.json
```

- Rotation 디렉터리는 **실제 최적화 조건**을 나타낸다. GPTQ용이면 W16이다.
- PTQ 로그/metadata 파일명은 **최종 target 조건**을 나타낸다. W4 평가이면 W4이며, rotation on/off·GPTQ/RTN·Had·실제 R3/R4·steps·학습 W·seed도 표시한다.
- Rotation off는 학습하지 않으므로 파일명에 steps/rotW를 넣지 않는다. PTQ metadata의 `max_steps`, `rotation_w_bits`, `learning_rate`, `rotation_checkpoint`는 null이다.
- PTQ 로그 첫 부분에도 target bit와 학습 bit, rotation/Had/실제 R3/R4, seed, 학습 step과 checkpoint 경로를 기록한다.

캐시 해시에는 전체 모델 식별자, **최적화 명령**의 모든 인자, 학습 관련 Python 소스 내용, 주요 패키지 버전이 포함된다. 캐시 디렉터리명은 실제 R3/R4 설정을 유지하므로 이전과 실행 명령이 같으면 기존 캐시를 재사용할 수 있다. 로컬 모델은 절대 경로와 내부 파일의 상대 경로·크기·수정 시각도 포함한다. A/K/V, R3/R4, max_steps, seed, learning_rate, batch size, sequence length, clipping/group 설정이 달라지면 캐시가 분리된다.

GPTQ의 target W는 캐시 키에 넣지 않는다. target W4/W8은 동일한 W16 학습 캐시를 공유하지만 PTQ 결과는 별도로 기록한다. RTN W4 최적화와 GPTQ W16 최적화는 서로 다른 캐시다. 실제 학습 명령이 완전히 같은 경우에만 GPTQ/RTN 사이에도 공유할 수 있다.

Rotation `metadata.json`은 기존 `spec`과 `checkpoint_sha256`을 유지하고, 읽기 쉬운 `rotation_config`를 추가한다. 이 객체에는 model, rotation_w_bits, A/K/V, rotation, had, R3/R4, max_steps, learning_rate, seed를 기록한다. 여러 target W/quantizer가 공유할 수 있으므로 특정 target을 checkpoint 자체의 조건으로 기록하지 않는다.

대신 각 **PTQ 로그 옆의 `.metadata.json`**에 다음과 같이 해당 실험의 전체 조건을 기록한다.

```json
{
  "model": "meta-llama/Llama-3.2-1B",
  "rotation_w_bits": 16,
  "target_w_bits": 4,
  "a_bits": 4,
  "k_bits": 16,
  "v_bits": 16,
  "quantizer": "gptq",
  "rotation": true,
  "had": true,
  "r3": false,
  "r4": true,
  "max_steps": 10,
  "learning_rate": 1.5,
  "seed": 0
}
```

실제 파일에는 checkpoint 경로·SHA-256 및 최종 실행 명령도 포함한다. 이 metadata는 실행 직전에 작성하는 **실험 설정 기록**이며 평가 성공을 보장하는 표시는 아니다. PPL과 실패 여부는 로그에서 확인한다. 동일 설정과 checkpoint의 재평가 로그/metadata는 덮어쓴다.

## 캐시 검증과 기존 결과

학습 성공 및 비어 있지 않은 R.bin 생성 후에만 완료 메타데이터를 게시한다. 재사용 시 설정과 파일 SHA-256을 확인한다. 동시 실행은 잠금으로 동일 캐시의 중복 학습을 방지한다. 강제 재학습 실패 시 기존 완료 캐시는 보존하며 `optimize.log`는 가장 최근 학습 시도를 기록한다. 중간 Trainer 체크포인트 저장은 끄고 최종 R.bin을 보관한다.

기존 파일은 삭제하지 않는다. 메타데이터 없는 R.bin은 자동 재사용하지 않으며, 기존 W4 학습 캐시는 새 GPTQ W16 학습 요청과 일치하지 않으므로 사용하지 않는다. 원격 revision 변경은 조회하지 않는다. 같은 원격 ID의 가중치가 바뀌었거나 로컬 파일의 크기·수정 시각을 유지한 채 내용을 바꿨다면 `FORCE_ROTATION=1`로 갱신한다.

이전에 rotation optimization에서도 target W4를 사용한 GPTQ 결과는 pipeline 검증 기록으로 보관한다. 논문 GPTQ baseline과 직접 비교하는 결과로 사용하지 않으며, 새 W16 최적화 조건으로 다시 측정한다. 현재 10-step 결과 역시 최종 100-step baseline과 구분한다.

## 검증

```bash
bash -n scripts/run_ptq.sh
python -m unittest discover -s tests -v
```

가짜 torchrun으로 GPTQ 학습 W16/평가 target W 분리, A/K/V 유지, RTN 분기, No Rotation, 파일명/metadata, 캐시 공유·분리·실패 복구를 검증한다. Had on/off와 K4/K8/K16의 조합을 확인하고, CPU Hadamard 대체 연산으로 R3/R4 분기와 K 양자화도 확인한다. 실제 GPU 학습·CUDA 커널·전체 모델 perplexity 검증은 포함하지 않는다.
