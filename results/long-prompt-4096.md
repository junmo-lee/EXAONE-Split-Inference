# 4,096토큰 프롬프트에서의 분할 추론 비교

동일한 RTX 4090 인스턴스에서 공식 GPU-only NF4와 CPU 4개·GPU 60개 레이어 TCP 분할을 각각 3회 측정했다. 입력은 채팅 템플릿을 포함해 정확히 4,096토큰인 합성 로그다. 끝에 있는 색을 묻고 최대 12개 토큰을 greedy 방식으로 생성했다. 모든 실행이 `The access color is amber.`를 출력했고 EOS를 포함해 7토큰을 생성했다. 두 경로의 토큰 ID도 일치했다.

| 구성 | 프리필 초 | TTFT 초 | 디코드 토큰/s | 장치 VRAM 표본 최대 MiB |
|---|---:|---:|---:|---:|
| GPU-only NF4 | 2.097 ± 0.002 | 2.097 ± 0.002 | 12.779 ± 0.044 | 21,614 |
| CPU 4/GPU 60, TCP 1 ms·10 Gbps | 158.679 ± 42.501 | 158.721 ± 42.506 | 3.650 ± 0.082 | 17,206 |

±는 표본 표준편차다. TTFT는 모델 로딩, 토큰화, TCP 세션 초기화 이후 첫 토큰 선택까지다. 공식 프리필에는 GPU LM Head가 포함되며 분할 프리필은 GPU hidden state 반환까지다. 디코드 처리율은 두 번째 토큰부터 계산했다.

- TCP 분할의 TTFT는 GPU-only의 75.7배였으며, 디코드 처리량은 28.6%였다.
- 분할 프리필 단계에서 CPU 계산과 최종 헤드가 156.54초, GPU 계산이 1.97초 걸렸다. 실제 루프백 TCP·프레임 처리의 추정 잔여 시간은 163.1 ms였다. PCIe 복사는 평균 14.0 ms였다.
- 프리필 요청은 41,943,120 B를 송신했고 응답은 첫 반복에서 10,482 B였다. 시뮬레이터는 이 왕복에 평균 34.65 ms를 주입했다. 처리량 제한은 애플리케이션에서 지연을 가산하는 방식으로 모사했다.
- 디코드 토큰당 CPU 계산은 평균 196.1 ms, GPU 계산은 75.6 ms였다. 이 하드웨어와 소프트웨어 구성에서는 CPU 계산이 프리필 지연의 대부분을 차지했다.

## 재현과 측정 조건

초기 TCP 시도는 CPU 스레드 8개에서 180초 제한을 넘겨 서버가 프리필 요청을 받기 전에 연결을 닫았다. 이 시도의 GPU-only 측정 3회는 완료되어 그대로 사용했다. TCP 경로는 CPU 스레드 24개와 서버 대기 시간 600초로 재실행했다. GPU-only는 4,096토큰 워밍업, TCP는 짧은 질문으로 워밍업했다. 앞선 23토큰 벤치마크의 CPU 스레드는 8개였으므로, 두 실험의 차이에 스레드 설정과 워밍업 차이가 함께 포함된다.

```bash
OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 python -u -m experiments.long_prompt \
  --dir /root/exaone-model --output results/long-prompt-rerun.json \
  --reuse-gpu-results results/long-prompt-4096.json --threads 24 --short-tcp-warmup
```

[전체 측정 JSON](long-prompt-4096-retry.json) · [초기 GPU-only 원시 JSON](long-prompt-4096.json)
