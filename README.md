# 🚀 Real-Time Object Edge Detection and Size Measurement System

## 프로젝트 개요
본 프로젝트는 박스 형태의 물체를 자동으로 감지하고 크기를 측정하는 프로그램을 개발하여 물류 프로세스를 개선하는 것을 목표로 합니다.

---

## 🛠️ 아키텍처  

1. **CAM**: 실시간 영상 입력 (BGR Matrix)
2. **색상 인식**: HSV 마스크 기반 박스 색상 인식
3. **객체 크기 측정**: 객체 외곽선과 Convex Hull을 사용한 경계 검출
4. **Capture 기능**: 버튼을 통해 이미지 캡처
5. **문자 인식**: Pytesseract로 텍스트 감지
6. **결과 표시**: 화면에 실시간으로 객체 경계 및 크기 표시

---

## 🧠 사용 기술  

- **OpenCV (cv2)**: 실시간 영상 처리 및 객체 감지  
- **PySide6 (Qt)**: GUI 및 사용자 인터페이스  
- **Numpy**: 고성능 수치 계산  
- **Pytesseract**: 이미지에서 문자 인식  
- **Scikit-learn (SGDClassifier)**: 숫자 인식 머신러닝 모델  

---

## 📝 주요 기능

1. **객체 인식 (HSV 마스킹)**
   - BGR 이미지를 HSV로 변환
   - 특정 색상 범위로 객체 마스킹
   - 객체 외곽선 감지 및 최대 면적 계산

2. **엣지 검출 (Sobel 필터)**
   - 그레이스케일 변환 및 Gaussian Blur 적용
   - Sobel 필터로 엣지 검출
   - 이진화로 경계 강조

3. **객체 크기 측정**
   - 외곽선 감지 및 경계 상자 그리기
   - 픽셀-센티미터 변환 비율로 크기 측정

---

## 💻 코드 예제

### **엣지 검출 (Sobel 필터) 예제**

```python
import cv2
import numpy as np

# 이미지 불러오기
image = cv2.imread('input_image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# GaussianBlur로 노이즈 제거
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Sobel 필터를 사용한 엣지 검출
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # X축 방향 경계 검출
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # Y축 방향 경계 검출

# 절댓값을 취해 경계 강조
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_combined = np.uint8(sobel_combined)

# 결과 출력
cv2.imshow('Sobel Edge Detection', sobel_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 🌟 핵심 기능
- 실시간 객체 검풀 및 크기 측정
- 숫자 인식: 객체 내 숫자 감지 및 결과 출력
- GUI 통합: 사용자 인터페이스를 통해 실시간 결과 표시

---

## 🖥️ 개발 환경
- Python 3.x
- OpenCV
- PySide6
- Scikit-learn
- Numpy
- Pytesseract

---

## 📊 결과
- 실시간 객체 경계선 검출
- 센티미터 단위의 정확한 객체 크기 측정
- 숫자 인식 및 화면 표시

---

## 🤔 어려웠던 점 및 해결 방법
1. 카메라 한계:

	- 다수의 객체를 동시에 정확히 인식하기 어려움
	- 해결: 추가 센서 도입 고려

2. 조명과 각도 문제:

	- 환경 변화에 따라 인식률 저하
	- 해결: 일정한 환경 유지 및 센서 개선

3. 시스템 유연성:

	- 새로운 형태의 객체 대응 부족
	- 해결: 모듈화 설계로 유연성 확보

---

## 📝 느낀 점
- 물류 자동화는 여러 요소가 복합적으로 연결됨.
- 작은 오류도 시스템 전체에 큰 영향을 미칠 수 있다는걸 알았음.
- 현장 피드백과 지속적인 개선이 필요.

---

## 🌟 팀원
- 고의근
- 김도하
- 정태현

