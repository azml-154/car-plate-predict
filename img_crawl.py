from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

#웹 열기(이미지 검색창)

driver = webdriver.Chrome()
driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&authuser=0&ogbl")

#검색
elem = driver.find_element_by_name("q")  #검색창
elem.send_keys("자동차")                 #검색
elem.send_keys(Keys.RETURN)             #enter key 전송


SCROLL_PAUSE_TIME = 1

#image 다운 전 스크롤 쭉 내리기
# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")#javascript 실행(브라우저 높이 탐색)
while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") #브라우저 끝까지 스크롤 내리기
    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)  #대기 (로딩 시간)
    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight") #로딩 후 브라우저 높이 구하기
    if new_height == last_height:  #높이 같다면 계속 탐색
        try:
            driver.find_element_by_css_selector(".mye4qd").click()  #결과 더보기 클릭
        except:
            break  #오류난 경우 반복문 빠져나가기
    last_height = new_height
    
    
    
#image download

images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")  #작은 이미지들의 class명을 탐색
count = 1
for image in images:
    try:
        image.click()   #img click
        time.sleep(2)   #img 로딩까지 기다리는 시간
        imgUrl = driver.find_element_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div/div[2]/a/img').get_attribute("src")  #이미지 url 받기
        opener=urllib.request.build_opener()
        opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/ 89.0.4389.90 Safari/537.36')]
        urllib.request.install_opener(opener) #다운
        urllib.request.urlretrieve(imgUrl, str(count) + ".jpg") #저장
        count = count + 1
    except:
        pass

driver.close()