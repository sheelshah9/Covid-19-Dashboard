import xml.etree.ElementTree as ET
import requests
import predict
import time

default_id = ""

while True:
    url = "https://github.com/CSSEGISandData/COVID-19/commits/master.atom"
    content = requests.get(url)

    xml = ET.fromstring(content.content)

    for child in xml.findall("{http://www.w3.org/2005/Atom}entry"):
        id = child.find("{http://www.w3.org/2005/Atom}id").text.strip()
        title = child.find("{http://www.w3.org/2005/Atom}title").text.strip()
        if title.lower().strip() == "automated update":
            if default_id != id:
                default_id = id
                predict.run_predictions()
            break

    time.sleep(1800)
