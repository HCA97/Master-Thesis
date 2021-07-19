import requests
import urllib.request
from urllib.error import HTTPError
import cv2
import os
from bs4 import BeautifulSoup


def get_car(url, folder, car_name, view="top"):
    response = requests.get(url)
    if response.ok:
        soup = BeautifulSoup(response.text, "html.parser")
        div_gallery = None
        for div in soup.find_all("div"):
            try:
                if "gallery-0" in div["id"]:
                    div_gallery = div
                    break
            except KeyError:
                pass

        if div_gallery:
            imgs = div_gallery.find_all("img")

            urls = [img["src"] for img in imgs]
            views = [img["alt"].split("-")[-1].lower() for img in imgs]

            try:
                idx = views.index(view)
                os.makedirs(folder, exist_ok=True)
                car_name = car_name.replace("/", "_")
                name = os.path.join(folder, car_name + "_" + view + ".png")
                urllib.request.urlretrieve(urls[idx], name)

                img = cv2.imread(name)
                if img is None:
                    raise RuntimeError
                cv2.imwrite(name, img)
            except ValueError:
                print(f"No {view} picture.")
            except HTTPError:
                print(f"404 Error on -> {urls[idx]}")
            except RuntimeError:
                print(f"{car_name} - {view} file is empty")


url = 'https://gta.fandom.com/wiki/Vehicles_in_GTA_Online'
# url = "https://gta.fandom.com/wiki/Vehicles_in_GTA_V"
save_path = "../potsdam_data/gta_cars_online"
response = requests.get(url)

if response.ok:
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find_all("table")[1]

    ths = table.find_all("th")
    tds = table.find_all("td")

    exclude = ["Motorcycles",
               "Off-Road",
               "Open Wheel",
               "Cycles",
               "Military",
               "Boats",
               "Planes",
               "Helicopters",
               "Super",
               "Utility",
               "Sports Classics",
               "Emergency",
               "Commercial",
               "Industrial",
               "Service",
               "Sports",
               "Muscle"]

    for th, td in zip(ths, tds):
        skip = False
        for e in exclude:
            if e in str(th):
                skip = True
        if not skip:
            links = td.find_all("a")

            for link in links:
                try:
                    href, title = link["href"], link["title"]

                    new_url = "https://gta.fandom.com" + href
                    get_car(new_url, save_path, title)
                except KeyError:
                    print(f"{href}, {title}")
