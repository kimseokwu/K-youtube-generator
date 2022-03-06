# Description

KoGPT2 모델에 K-유튜브, 속된 말로 '국뽕 유튜브'의 제목을 학습시켜 새로운 K-유튜브 제목을 생성해보는 토이프로젝트입니다.

이 프로젝트는 2022.02.01 부터 2022.03.12(예정)까지 진행 되었습니다.

자세한 프로젝트 설명은 [블로그 링크](https://littledatascientist.tistory.com/92?category=981840)로 들어가시면 보실 수 있습니다.

# Data

-  `Sellenium`을 이용해 크롤링한 K-유튜브 채널 23개의 영상 9,261개에 대한 데이터
- 수집된 영상의 업로드 기간: 2013.08.11 - 2022-02-13
- __Fields__
  - url(string): 해당 영상의 URL
  - vid_name(string): 해당 영상의 제목
  - hashtags(string): 해당 영상에 포함된 해시태그들(띄어쓰기로 분리)
  - views(integer): 해당 영상의 조회수
  - likes(string): 해당 영상의 좋아요 수
  - upload_date(string): 해당 영상이 업로드 된 날짜
  - channel_name(string): 해당 영상이 올라온 유튜브 채널의 이름
  - channel_subs(string): 해당 영상이 올라온 유튜브 채널의 구독자 수
- __데이터 샘플 이미지__

![데이터 샘플](https://github.com/kimseokwu/K-youtube-generator/blob/main/image/data_example.png?raw=true)

# Output

![](https://github.com/kimseokwu/K-youtube-generator/blob/main/image/output_example.png?raw=true)

# Reference

- [KoGPT2 (한국어 GPT-2) Ver 2.0](https://github.com/SKT-AI/KoGPT2)
- [KoGPT2-FineTuning](https://github.com/gyunggyung/KoGPT2-FineTuning)
