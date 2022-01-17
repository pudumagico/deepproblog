import re

d2s = {
    "0": "LaGraMeCy",
    "1": "LaGraMeSp",
    "2": "LaGraMeCu",
    "3": "LaGraRuCy",
    "4": "LaGraRuSp",
    "5": "LaGraRuCu",
    "6": "LaBlMeCy",
    "7": "LaBlMeSp",
    "8": "LaBlMeCu",
    "9": "LaBlRuCy",
    "10": "LaBlRuSp",
    "11": "LaBlRuCu",
    "12": "LaBrMeCy",
    "13": "LaBrMeSp",
    "14": "LaBrMeCu",
    "15": "LaBrRuCy",
    "16": "LaBrRuSp",
    "17": "LaBrRuCu",
    "18": "LaYeMeCy",
    "19": "LaYeMeSp",
    "20": "LaYeMeCu",
    "21": "LaYeRuCy",
    "22": "LaYeRuSp",
    "23": "LaYeRuCu",
    "24": "LaReMeCy",
    "25": "LaReMeSp",
    "26": "LaReMeCu",
    "27": "LaReRuCy",
    "28": "LaReRuSp",
    "29": "LaReRuCu",
    "30": "LaGreMeCy",
    "31": "LaGreMeSp",
    "32": "LaGreMeCu",
    "33": "LaGreRuCy",
    "34": "LaGreRuSp",
    "35": "LaGreRuCu",
    "36": "LaPuMeCy",
    "37": "LaPuMeSp",
    "38": "LaPuMeCu",
    "39": "LaPuRuCy",
    "40": "LaPuRuSp",
    "41": "LaPuRuCu",
    "42": "LaCyMeCy",
    "43": "LaCyMeSp",
    "44": "LaCyMeCu",
    "45": "LaCyRuCy",
    "46": "LaCyRuSp",
    "47": "LaCyRuCu",
    "48": "SmGraMeCy",
    "49": "SmGraMeSp",
    "50": "SmGraMeCu",
    "51": "SmGraRuCy",
    "52": "SmGraRuSp",
    "53": "SmGraRuCu",
    "54": "SmBlMeCy",
    "55": "SmBlMeSp",
    "56": "SmBlMeCu",
    "57": "SmBlRuCy",
    "58": "SmBlRuSp",
    "59": "SmBlRuCu",
    "60": "SmBrMeCy",
    "61": "SmBrMeSp",
    "62": "SmBrMeCu",
    "63": "SmBrRuCy",
    "64": "SmBrRuSp",
    "65": "SmBrRuCu",
    "66": "SmYeMeCy",
    "67": "SmYeMeSp",
    "68": "SmYeMeCu",
    "69": "SmYeRuCy",
    "70": "SmYeRuSp",
    "71": "SmYeRuCu",
    "72": "SmReMeCy",
    "73": "SmReMeSp",
    "74": "SmReMeCu",
    "75": "SmReRuCy",
    "76": "SmReRuSp",
    "77": "SmReRuCu",
    "78": "SmGreMeCy",
    "79": "SmGreMeSp",
    "80": "SmGreMeCu",
    "81": "SmGreRuCy",
    "82": "SmGreRuSp",
    "83": "SmGreRuCu",
    "84": "SmPuMeCy",
    "85": "SmPuMeSp",
    "86": "SmPuMeCu",
    "87": "SmPuRuCy",
    "88": "SmPuRuSp",
    "89": "SmPuRuCu",
    "90": "SmCyMeCy",
    "91": "SmCyMeSp",
    "92": "SmCyMeCu",
    "93": "SmCyRuCy",
    "94": "SmCyRuSp",
    "95": "SmCyRuCu"
}

s2l = {
    "sizes": {
        "La": "large",
        "Sm": "small"
    },
    "shapes": {
        "Cy": "cylinder",
        "Sp": "sphere",
        "Cu": "cube"
    },
    "materials": {
        "Me": "metal",
        "Ru": "rubber"
    },
    "colors": {
        "Gra": "gray",
        "Bl": "blue",
        "Br": "brown",
        "Ye": "yellow",
        "Re": "red",
        "Gre": "green",
        "Pu": "purple",
        "Cy": "cyan"
    }
}

lst = []

for key in d2s:
    lst.append(d2s[key])

real_list = []

for x in lst:
    Size, Color, Material, Shape = re.sub(r"([A-Z])", r" \1", x).split()
    real_list.append(f'obj(B,{s2l["shapes"][Shape]},{s2l["sizes"][Size]},{s2l["colors"][Color]},{s2l["materials"][Material]},X1,Y1,X2,Y2)')

print(str(f'{real_list}').replace("'", '"'))
