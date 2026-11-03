from agents.bcf import BCFAgent
from agents.larbrid import LARBRIDAgent
from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.larbrid_i import LARBRIDIAgent
from agents.larbrid_i_img import LARBRIDIImgAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent
from agents.larbrid_o2o import LARBRIDO2OAgent

agents = dict(
    bcf=BCFAgent,
    larbrid=LARBRIDAgent,
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    larbrid_i=LARBRIDIAgent,
    larbrid_i_img=LARBRIDIImgAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
    larbrid_o2o=LARBRIDO2OAgent,
)
