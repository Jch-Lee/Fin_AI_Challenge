# GitFlow 기반 워크플로우

# 핵심 브랜치 역할
- main 브랜치: 가장 안정적인 최종 버전을 관리합니다. 에픽(Epic) 완료와 같은 주요 마일스톤 달성 시에만 업데이트됩니다.

- develop 브랜치: 개발의 중심이 되는 브랜치입니다. 모든 기능 개발이 완료되면 이 브랜치로 통합(merge)됩니다. 항상 다음 릴리즈를 준비하는 최신 개발 버전입니다.

- feature 브랜치: 각각의 스토리를 개발하기 위한 브랜치입니다. 항상 develop 브랜치에서 생성됩니다.

# 작업 과정 (Step-by-Step)

## 브랜치 생성 (Create Branch) 🌿

- 새로운 스토리 작업을 시작하기 전, develop 브랜치에서 최신 코드를 pull 받습니다.

- develop 브랜치에서 새로운 feature 브랜치를 생성합니다.

- 예시: feat/story-1.1/init-project

## 개발 및 커밋 (Develop & Commit) 💻

- 생성한 feature 브랜치에서 코드를 작성하고, 작은 단위로 커밋합니다.

## Pull Request (PR) 생성 📤

- 기능 개발이 완료되면, develop 브랜치로 병합해달라는 Pull Request(PR)를 생성합니다.

- 코드 리뷰 (Code Review) 🧐

- 다른 팀원이 PR을 리뷰하고 피드백을 주고받습니다.

## develop 브랜치에 병합 (Merge to Develop) 🤝

- 코드 리뷰가 완료되면, feature 브랜치를 develop 브랜치에 병합합니다.

- 브랜치 삭제 (Clean Up) 🗑️

- develop에 병합이 완료된 feature 브랜치는 삭제합니다.

## 마일스톤 릴리즈 (Milestone Release) 🎉

- 하나의 에픽(Epic)이 완료되는 등, 중요한 개발 마일스톤에 도달하면, develop 브랜치를 main 브랜치로 병합하여 안정적인 버전을 만듭니다. 이 때 v1.0과 같은 태그(tag)를 생성하여 버전을 관리할 수 있습니다.

## 시각화된 흐름
      (Tag: v1.0)
main <-----(Merge for Release)---- develop
 ^                                /
 |                               /
 |                             / (Merge PR)
 +--------------------------- develop <---- feature/story-1.2
                             /
                           / (Merge PR)
 develop <---- feature/story-1.1