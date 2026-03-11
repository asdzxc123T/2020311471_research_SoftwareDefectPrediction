## Docker 재현성 가이드

이 문서에서는 제 실험 환경을 Docker로 재현하시는 방법과, Joern·Defects4J 같은 외부 도구를 제가 어떻게 붙여 두었는지 정리했습니다.

### 기본(Trainer)

Python 실험은 `trainer` 컨테이너에서 돌리도록 했습니다. 리포지토리 루트에서 아래 명령을 실행하시면 됩니다.

```bash
docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml exec trainer pip install -e .
```

### Joern(선택)

CPG 기반 구조 정보는 Joern으로 뽑도록 설계했고, Joern은 별도 서비스로 두어 필요할 때만 켜도록 했습니다. 제안서대로 초기 실험에서는 **AST 기반 경량 그래프**를 먼저 쓰고, 이후 단계에서 Joern/CPG 기반 분석을 확장하는 구성을 염두에 두고 있습니다. 혹시 사용해보실 계획이 있으시다면, Joern을 쓰실 때는 버전(release tag), 대상 언어·입력 규모, 산출물 경로를 고정해 두시는 것을 권장합니다.

`docker-compose.yml`에 `joern` 서비스가 들어 있어 있고, `profiles: ["optional"]`로 두어서 필요할 때만 올리도록 했습니다.

```bash
docker compose -f docker/docker-compose.yml --profile optional up -d joern
docker compose -f docker/docker-compose.yml exec joern joern --version
```

`joernio/joern:latest`는 환경에 따라 제공이 바뀔 수 있으니, 실제 실험 시에는 사용하시는 Joern 버전(태그)을 고정해 두시는 것이 좋습니다.

### Defects4J(선택)

Defects4J는 의존성이 많아서, 제가 스크립트 자리만 마련해 두었고 버전·경로는 사용하시는 쪽에서 정하시면 됩니다. Java 버전, perl, 빌드 도구(gradle/maven/ant), Defects4J 버전(tag)과 설치 경로를 정해 두고 쓰시면 재현이 수월합니다.

