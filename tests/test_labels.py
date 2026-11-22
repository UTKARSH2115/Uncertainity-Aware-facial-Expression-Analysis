import numpy as np
from m3.data.labels import process_row, FER2PLUS
def test_rules():
    r=process_row([0,10,0,0,0,0,0,0,0,0],"a.png"); assert r["keep"] and r["y"]==1 and not r["tied"] and r["p"][1]==1
    assert process_row([4,4,2,0,0,0,0,0,0,0],"a")["tied"]
    assert process_row([2,0,0,0,0,0,0,0,5,3],"a")["reason"]=="F3"
    assert process_row([5,0,0,0,0,0,0,0,0,0],"")["reason"]=="F1"
    assert abs(process_row([3,4,3,0,0,0,0,0,0,0],"a")["p"].sum()-1)<1e-6
def test_map():
    assert FER2PLUS=={0:4,1:5,2:6,3:1,4:3,5:2,6:0}
