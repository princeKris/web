import os
import re
import time
from scapy.all import *
cou=0
def ch(ipp):
  te="hi"
  ip=ipp
  ext = "(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})"
  #re.findall(extract_mac_address_pattern, 'Unknown error in node 00:00:5e:00:53:af. Terminating.')
  with os.popen(f'nmap -sP -PR {ip}') as f:
    data=f.read()
    #print(data)
    for line in re.findall(ext,data):
      lin=line.replace(":","-")
      #print(lin.lower())
      te=lin.lower()
      return te

def chip(ipp):
  te="hi"
  ip=ipp
  request = Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(pdst=ip)
  ans, unans = srp(request, timeout=2, retry=1)
  for sent, received in ans:
    line=received.hwsrc
    lin=line.replace(":","-")
    te=lin.lower()
    return te


def chh():
  global cou
  with os.popen('arp -a') as f:
    data = f.read()
    for line in re.findall(r'([-.0-9]+)\s+([-0-9a-f]{17})\s+(\w+)',data):
      #print(line)
      if(line[2]=='dynamic'):
        getmac=chip(line[0])
        if(getmac is None):
          break
        if(line[1] != getmac):
          cou=cou+1
          print(f"{line[1]} - {getmac}")
          print(f"[{cou}] under attack")
        

while(True):          
  chh()
  time.sleep(2)
