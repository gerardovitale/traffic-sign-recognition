WIDTH = 600
SCALE = 2
STEP = 10
WINDOW_SIZE = (180,180)
INPUT_SIZE = (180, 180)
MODEL_PATH = 'outputs/models/model_2.h5' 
IMAGE_DIR = 'dataset/images-for-recognition'
IMAGE_PATH = 'dataset/images-for-recognition/European-road-signs.jpg'
# European-road-signs
# german-traffic-signs-for-bikes


CLASS_ID = [
    '0', '1', '10', '11', '12', '13', '14', 
    '15', '16', '17', '18', '19', '2', '20', 
    '21', '22', '23', '24', '25', '26', '27', 
    '28', '29', '3', '30', '31', '32', '33', 
    '34', '35', '36', '37', '38', '39', '4', 
    '40', '41', '42', '5', '6', '7', '8', 
    '9', 'no-traffic-sign'
    ]

CLASS_NAMES = {
    '0': 'Limit speed 20 km/h',
    '1': 'Limit speed 30 km/h',
    '2': 'Limit speed 50 km/h',
    '3': 'Limit speed 60 km/h',
    '4': 'Limit speed 70 km/h',
    '5': 'Limit speed 80 km/h',
    '6': 'End of limit speed 80 km/h',
    '7': 'Limit speed 100 km/h',
    '8': 'Limit speed 120 km/h',
    '9': 'Overtaking prohibited',
    '10': 'Overtaking prohibited for trucks',
    '11': 'Warning for a crossroad side roads on the left and right',
    '12': 'Begin of a priority road',
    '13': 'Give way to all drivers (yield)',
    '14': 'Stop and give way to all drivers (stop)',
    '15': 'Entry prohibited',
    '16': 'Trucks prohibited',
    '17': 'Entry prohibited (road with one-way traffic)',
    '18': 'Warning for a danger with no specific traffic sign',
    '19': 'Warning for a curve to the left',
    '20': 'Warning for a curve to the right',
    '21': 'Warning for a double curve, first left then right',
    '22': 'Warning for a bad road surface',
    '23': 'Warning for a slippery road surface',
    '24': 'Warning for a road narrowing on the right',
    '25': 'Warning for roadworks',
    '26': 'Warning for a traffic light',
    '27': 'Warning for pedestrians',
    '28': 'Warning for children',
    '29': 'Warning for cyclists',
    '30': 'Warning for snow',
    '31': 'Warning for crossing deer',
    '32': 'Unknown',
    '33': 'Turning right mandatory',
    '34': 'Turning left mandatory',
    '35': 'Driving straight ahead mandatory',
    '36': 'Driving straight ahead or turning right mandatory', 
    '37': 'Driving straight ahead or turning left mandatory',
    '38': 'Passing right mandatory',
    '39': 'Passing left mandatory',
    '40': 'Mandatory direction of the roundabout',
    '41': 'End of the overtaking prohibition',
    '42': 'End of the overtaking prohibition for trucks',
    'no-traffic-sign': 'Not a traffic sign'
}
