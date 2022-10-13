import matplotlib.pyplot as plt
import numpy as np 
from collections import Counter 
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator



a = {'#childsexualabuse': 1, '#globalwellnessday': 1, '#wellnessday': 1, '#Business101': 1, '#huntervalley': 1, '#wombats': 1, '#wollombi': 1, '#UpperCoomera': 1, '#macular': 1, '#groups': 1, '#Kenilworth': 1, '#MePoo': 1, '#AmberHeardlsAnAbuser': 1, '#AmberHeardlsALiar': 1, '#DAEN': 1, '#Glenella': 1, '#scottyfromcorruption': 1, '#alien': 1, '#dalle': 1, '#aiart': 1, '#dallemini': 1, '#overturncitizensunited': 1, '#PeterDutton': 1, '#Lettuce': 1, '#Iceberg': 1, '#gameoncancer': 1, '#Carseldine': 1, '#abloodygreatnightout': 1, '#SayNoToWar': 1, '#Classic100': 1, '#GregInglis': 1, '#Dalby': 1, '#MountChalmers': 1, '#Margate': 1, '#BlueMountainHeights': 1, '#fluseason': 1, '#OurTaxesAtWork': 1, '#LNPStuffUp': 1, '#Allenview': 1, '#EvertonHills': 1, '#HollowaysBeach': 1, '#23': 1, '#NRLTitansSouths': 1, '#classic100': 1, '#Advancetown': 1, '#aec6': 1, '#Swanbank': 1, '#MountCootha': 1, '#Insider': 1, '#LowDoseCT': 1, '#1of2': 1, '#aflfreohawks': 1, '#histmed': 1, '#aftercare': 1, '#mindcontrol': 1, '#yvonnemcclaren': 1, '#australianbush': 1, '#sideeffect': 1, '#KnowMore': 1, '#StopWarOnTigray': 1, '#BBSC': 1, '#AFLFreoHawks': 1, '#TNTABC': 1, '#publishedauthor': 1, '#Hardwork': 1, '#AFLLionsSaints': 1, '#NSWpoll': 1, '#Morriscum': 1, '#ENGvsNZ': 1, '#AFCU23': 1, '#Dauphine': 1, '#TrashDiscoLive': 1, '#CutDownTransmission': 1, '#SARvHAR': 1, '#AlbofromMarketing': 1, '#AlboFromPhotoOps': 1, '#wesaytrans': 1, '#wesaygay': 1, '#leptomeningealdisease': 1, '#enhertu': 1}
mask = np.array(Image.open('C:/Users/Sai Ram/OneDrive/RMIT/twitter/dahboard-final/dashboard-deploy/assets/hash.jpg'))



image_colors = ImageColorGenerator(mask)
wordcloud = WordCloud(width=800, height=200, background_color="rgba(0, 0, 0, 0)", mask=mask
                     ).generate_from_frequencies(a)

# Display the generated image:
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.savefig('wordcloud.png', facecolor='k', bbox_inches='tight')
plt.show()

