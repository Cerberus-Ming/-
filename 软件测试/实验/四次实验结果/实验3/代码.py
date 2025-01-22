from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains


def main():
    # 加载驱动，请求网页，设置隐式等待时间
    driver = webdriver.Chrome()
    driver.get("https://nj.zu.anjuke.com/")
    driver.implicitly_wait(5)

    # 点击“租房”
    driver.find_element_by_xpath("/html/body/div[2]/div/ul/li[4]/a").click()

    # 鼠标停留在地址选择处
    chain = ActionChains(driver)
    implement = driver.find_element_by_xpath('//*[@id="switch_apf_id_5"]')
    chain.move_to_element(implement).perform()

    # 点击“南京”
    driver.find_element_by_xpath('//*[@id="city_list"]/dl[2]/dd/a[4]').click()

    # 点击“地铁找房”
    driver.find_element_by_xpath('/html/body/div[4]/ul/li[2]/a').click()

    # 点击“地铁”
    driver.find_element_by_xpath('/html/body/div[5]/div[2]/div[1]/span[2]/a[2]').click()

    # 点击“2号线”
    driver.find_element_by_xpath('/html/body/div[5]/div[2]/div[1]/span[2]/div/a[3]').click()

    # 点击“马群”
    driver.find_element_by_xpath('/html/body/div[5]/div[2]/div[1]/span[2]/div/div/a[24]').click()

    # 输入租金下限5000，上限8000，并点击确定
    driver.find_element_by_xpath('//*[@id="from-price"]').send_keys("5000")
    driver.find_element_by_xpath('//*[@id="to-price"]').send_keys("8000")
    driver.find_element_by_xpath('//*[@id="pricerange_search"]').click()

    # 选择“整租”
    driver.find_element_by_xpath('/html/body/div[5]/div[2]/div[4]/span[2]/a[2]').click()

    # 鼠标停留在房屋类型选择处
    chain = ActionChains(driver)
    implement = driver.find_element_by_xpath('//*[@id="condhouseage_txt_id"]')
    chain.move_to_element(implement).perform()

    # 点击“普通住宅”
    driver.find_element_by_xpath('//*[@id="condmenu"]/ul/li[2]/ul/li[2]/a').click()

    # 在搜索框中搜索“经天路”，点击“搜索”
    driver.find_element_by_xpath('//*[@id="search-rent"]').send_keys("经天路")
    driver.find_element_by_xpath('//*[@id="search-button"]').click()

    # 点击“视频看房”
    driver.find_element_by_xpath('//*[@id="list-content"]/div[1]/a[2]').click()


    # 点击第一个搜索到的房源
    driver.find_element_by_xpath('//*[@id="list-content"]/div[3]').click()


if __name__ == '__main__':
    main()
