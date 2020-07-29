import xml.etree.ElementTree as ET
import requests
import predict
import time

url = "https://github.com/CSSEGISandData/COVID-19/commits/master.atom"
content = requests.get(url)

xml = ET.fromstring(content.content)
default_id = ""

while True:
    for child in xml.findall("{http://www.w3.org/2005/Atom}entry"):
        id = child.find("{http://www.w3.org/2005/Atom}id").text.strip()
        title = child.find("{http://www.w3.org/2005/Atom}title").text.strip()
        if title=="automated update" and default_id!=id:
            default_id = id
            predict.run_predictions()
            break

    time.sleep(1800)