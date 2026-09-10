# PTQ 실험과 rotation 캐시

```bash
bash scripts/run_ptq.sh MODEL W_BITS A_BITS K_BITS V_BITS QUANTIZER ROTATION [R3] [R4]
```

- `QUANTIZER`: `gptq` 또는 `rtn` (PTQ 가중치 양자화 방식).
- `ROTATION`: R1/R2 최적화 및 적용 여부 (`on` / `off`).
- `R3`, `R4`: 각각 독립적으로 `on` / `off`. 최적화와 PTQ에 같은 값이 전달된다.
- R3 생략 시 K<16이면 on, K16이면 off. R4 생략 시 ROTATION을 따른다. 기존 7개 인자 호출의 rotation 동작을 유지한다.
- 명시적인 `R3=on`은 K16에서도 Q/K Hadamard를 실행한다. `R3=off`도 K4/K8 양자화는 유지한다.
- R4는 down_proj 가중치 변환과 입력 activation 변환을 함께 제어한다. ROTATION=off에서도 R4를 명시적으로 켤 수 있다.

```bash
# R1/R2만 사용, 10 step 최적화. 다음에 같은 명령을 실행하면 R.bin 재사용.
bash scripts/run_ptq.sh meta-llama/Llama-3.2-1B 4 8 4 4 gptq on off off

# 같은 R.bin으로 RTN 평가 (최적화 단계는 PTQ의 GPTQ/RTN 선택과 무관).
bash scripts/run_ptq.sh meta-llama/Llama-3.2-1B 4 8 4 4 rtn on off off

# 100 step 실험은 10 step 실험과 다른 캐시 사용.
MAX_STEPS=100 bash scripts/run_ptq.sh meta-llama/Llama-3.2-1B 4 8 4 4 gptq on off off

# R3/R4도 사용.
bash scripts/run_ptq.sh meta-llama/Llama-3.2-1B 4 8 4 4 gptq on on on

# 동일한 캐시 설정을 강제로 다시 학습.
FORCE_ROTATION=1 bash scripts/run_ptq.sh meta-llama/Llama-3.2-1B 4 8 4 4 gptq on off off
```

환경 변수:

| 변수 | 기본값 | 용도 |
|---|---|---|
| `MAX_STEPS` | `10` | Rotation 최적화 step 수 |
| `ROTATION_SEED` | `0` | 두 Python 실행에 전달하는 `--seed` |
| `FORCE_ROTATION` | `0` | `1`이면 유효한 캐시가 있어도 다시 최적화 |
| `RESULT_DIR` | `results` | PTQ 로그와 rotation 캐시 저장 루트 |

캐시는 다음 구조로 저장된다.

```text
results/rotation/
  Llama-3.2-1B_W4A8K4V4_steps10_r3-off_r4-off_<hash>/
    R.bin
    metadata.json
    optimize.log
```

해시에는 전체 모델 식별자(동일 basename의 다른 모델 구분), 최적화 명령의 모든 인자, 학습 관련 Python 소스 내용, 주요 패키지 버전이 포함된다. 따라서 W/A/K/V, R3/R4, max_steps, seed, learning_rate, batch size, sequence length, clipping/group 설정 등이 달라지면 캐시가 분리된다. 로컬 모델은 절대 경로와 내부 파일의 상대 경로·크기·수정 시각도 포함한다. 파일 이름에는 자주 비교하는 설정을 표시하고, `metadata.json`에는 전체 구분 정보를 기록한다.

`max_steps`는 반드시 구분한다. 학습량뿐 아니라 cosine learning-rate schedule도 바뀌므로 10-step 결과를 100-step 결과로 취급할 수 없다. 100-step 실행은 별도로 처음부터 학습하며, 10-step R.bin에서 resume하지 않는다. R.bin에는 R1/R2만 있고 optimizer/scheduler 상태가 없다.

학습이 성공하고 비어 있지 않은 R.bin이 생성된 뒤에만 완료 메타데이터를 게시한다. 재사용 시 설정과 파일 SHA-256을 확인한다. 같은 캐시를 요청한 동시 실행은 잠금으로 중복 학습을 방지한다. 강제 재학습이 실패하면 기존 완료 캐시는 보존한다. `optimize.log`는 가장 최근 학습 시도의 기록이다. 중간 Trainer 체크포인트 저장은 끄고 최종 R.bin을 보관한다.

기존의 메타데이터 없는 R.bin은 삭제하지 않으며 자동 재사용하지 않는다. 학습 설정과 완료 여부를 검증할 수 없기 때문이다. 원격 모델은 입력한 저장소 ID를 기준으로 구분하며 원격 revision 변경을 조회하지 않는다. 같은 ID의 원격 가중치가 바뀌었거나 로컬 파일의 크기·수정 시각을 유지한 채 내용을 바꿨다면 `FORCE_ROTATION=1`로 갱신한다.

PTQ는 캐시가 있어도 매번 실행한다. PTQ 로그는 캐시 식별자와 GPTQ/RTN·seed를 포함하며 동일 설정의 재평가 로그는 덮어쓴다. ROTATION=off에서는 rotation 학습/캐시 조회를 하지 않는다.

검증 명령:

```bash
bash -n scripts/run_ptq.sh
python -m unittest discover -s tests -v
```

테스트는 가짜 torchrun으로 캐시 생명주기를 확인하고, CPU Hadamard 대체 연산으로 rotation 분기와 K 양자화를 검증한다. 실제 GPU 학습·CUDA 커널·전체 모델 perplexity 검증은 포함하지 않는다.
