import csv
import math
import os
import warnings

import google_streetview.api
import numpy as np
import overpy
import pandas as pd
import plotly.express as px
import yaml

from lobe import ImageModel
from scipy.spatial import ConvexHull



"""
This warning will be displayed if no lamps have been detected at the end of the
process, in order to notify the user of the problem. 
"""


class NoLampsWarning(UserWarning):
    pass



"""
The "distance" function calculates the distance between two points
on the surface of a sphere using (lat,lng) coordinates and the radius 
of the sphere (6371 km in the case of the Earth).
"""
        

def distance(R1, R2, r = 6371):
    R1 = (math.pi/2 - math.radians(R1[0]), math.radians(R1[1]))
    R2 = (math.pi/2 - math.radians(R2[0]), math.radians(R2[1]))
    P1 = (math.cos(R1[1])*math.sin(R1[0]), math.sin(R1[1])*math.sin(R1[0]), math.cos(R1[0]))
    P2 = (math.cos(R2[1])*math.sin(R2[0]), math.sin(R2[1])*math.sin(R2[0]), math.cos(R2[0]))
    cosα = np.dot(P1, P2)
    if cosα >= 1:
        cosα = 1
    elif cosα <= -1:
        cosα = -1
    α = math.acos(cosα)
    distance = α*r
    return distance


"""
The "area" function approximates the area
of a rectangle on the surface of a sphere.
"""


def area(rectangle):
    b = (distance(rectangle[0], rectangle[1]) + distance(rectangle[2], rectangle[3]))/2
    h = (distance(rectangle[1], rectangle[2]) + distance(rectangle[3], rectangle[0]))/2
    return b*h


"""
The "bounding_box" function makes a rectangle surrounding all points in a given set
given its convex hull and a starting point in the hull. It uses (x,y) cartesian coordinates.
"""


def bounding_box(hull, i):
    
    P = hull[i]
    Q = hull[i+1]
    PQ = [Q[0]-P[0], Q[1]-P[1]]
    
    if PQ[0] == 0:
        θ = {True: math.pi/2, False: 3*math.pi/2}[PQ[1]>0]
    elif PQ[0] > 0:
        θ = math.atan(PQ[1]/PQ[0])
    else:
        θ = math.atan(PQ[1]/PQ[0]) + math.pi
        
    φ = 2*math.pi-θ
    rotated_coords = [(coords[0]*math.cos(φ)-coords[1]*math.sin(φ), coords[0]*math.sin(φ)+coords[1]*math.cos(φ)) for coords in hull]
    x = [coords[0] for coords in rotated_coords]
    y = [coords[1] for coords in rotated_coords]
    rotated_rectangle = [
        (min(x), min(y)),
        (min(x), max(y)),
        (max(x), max(y)),
        (max(x), min(y)),
    ]
    
    rectangle = [(coords[0]*math.cos(-φ)-coords[1]*math.sin(-φ), coords[0]*math.sin(-φ)+coords[1]*math.cos(-φ)) for coords in rotated_rectangle]
    
    return rectangle, area(rectangle), φ


"""
The "minimum_bounding_box" function uses the "bounding_box" function to create all possible bounding
boxes of a set of points and chooses the one with the lowest surface area.
"""


def minimum_bounding_box(points):
    
    hull = [points[i] for i in ConvexHull(points).vertices]
    hull.append(hull[0])
    
    bounding_boxes = []
    areas = []
    angles = []
    
    for i in range(len(hull)-1):
        
        box = bounding_box(hull, i)
        bounding_boxes.append(box[0])
        areas.append(box[1])
        angles.append(box[2])
     
    minimum = areas.index(min(areas))
    min_box = bounding_boxes[minimum]
    φ = angles[minimum]
    return min_box, φ


"""
The "OSMrequest" function makes a request to the OpenStreetMap database
using the Overpass API to obtain the polygon of the city. It also uses the 
Simplemaps World Cities database to find the central coordinate of the city, 
to eliminate faraway results. This polygon may be inaccurate, thus user 
will have to verify the result in the "polygon_verification" function.
"""


def OSMrequest(city, country, max_distance = 250):
    print('Downloading the polygon of the zone to study...')    
    SMWC = f'{path}worldcities.csv'
    cities = pd.read_csv(SMWC)
    cities = cities[cities['country'] == country]
    cities = cities[cities['city'] == city]
    cities = cities.reset_index()
    
    if len(cities):
        center_coords = (cities['lat'][0], cities['lng'][0])
    else:
        raise ValueError(f"{city}, {country} is not within the Simplemaps World Cities database.")
    
    api = overpy.Overpass()
    query = f"""
    [out:json];
    relation["type"="boundary"]["name"="{city}"];
    (._;>;);
    out body;
    """
    response = api.query(query)
    filtered_city = [i for i in response.nodes]
    coords_city = [(float(i.lat),float(i.lon)) for i in filtered_city]
    
    polygon = []
    for node in coords_city:
        if distance(center_coords, node) < max_distance:
            polygon.append(node)
            
    print('Done')
    return polygon


"""
The "polygon_verification" function exists to ensure that the OSM request gave back 
the wanted polygon. If that is not the case, the user will have to manually input
the polygon in the params.txt file.
"""


def polygon_verification(polygon):
    df = pd.DataFrame(polygon, columns = ('lat','lon'))
    fig = px.scatter_mapbox(df, lat='lat', lon='lon')
    fig.update_layout(mapbox_style = 'open-street-map')
    fig.show()
    
    ok = False
    while not ok:
        check = input('Is this polygon ok? [Y/n]\n').upper()
        if check in ['Y', 'YES']:
            ok = True
        elif check in ['N', 'NO']:
            raise ValueError("Please manually input the polygon in the params.txt file and change the method to 'coords'")
        
    

"""
The "autorisation" function is there to ensure that the user does
not accidentally make a request that is too expensive.
"""


def autorisation(shape):
    ok = False
    formatted = "{:.2f}"
    
    while not ok:       
        if type(shape[0]) != int or type(shape[1]) != int or shape[0] < 1 or shape[1] < 1:
            raise ValueError('Shape must be a list of two positive integers')
            
        amount = (shape[0]*shape[1])*0.028
        check = input(f'Shape is currently {shape} ({shape[0]*shape[1]} subzones), which means the process will cost up to $'+formatted.format(amount)+'. Is this ok? [Y/n]\n').upper()
        
        if check in ['Y', 'YES']:
            ok = True
        
        elif check in ['N', 'NO']:
            new_value_found = False    
            while not new_value_found:
                try:
                    shape_lat = int(input('Please input new values:\n'))
                    shape_lon = int(input())
                    shape = (shape_lat, shape_lon)
                    new_value_found = True
                
                except ValueError:
                    print('Values must be integers')
    
    return shape


"""
The "polygons" function separates the studied zone into shape[0]*shape[1] subzones of 
approximately equal area.
"""  


def polygons(polygon, shape, φ):

    print(f'Separating the studied zone in {shape[0]*shape[1]} subzones...')
    rotated_polygon = [(coords[0]*math.cos(φ)-coords[1]*math.sin(φ), coords[0]*math.sin(φ)+coords[1]*math.cos(φ)) for coords in polygon]
    
    rot_lat = [i[0] for i in rotated_polygon]
    rot_lng = [i[1] for i in rotated_polygon]
    
    if max(rot_lat) - min(rot_lat) >= max(rot_lng) - min(rot_lng):
        shape = (max(shape), min(shape))
    else:
        shape = (min(shape), max(shape))
    
    rot_zones = []
    zones = []
    
    lat_1 = min(rot_lat)
    for i in range(shape[0]):
        lon_1 = min(rot_lng)
        lat_2 = lat_1 + (max(rot_lat)-min(rot_lat))/shape[0]
        for j in range(shape[1]):
            lon_2 = lon_1 + (max(rot_lng)-min(rot_lng))/shape[1]
            rot_zones.append([
                (lat_1, lon_1),
                (lat_2, lon_1),
                (lat_2, lon_2),
                (lat_1, lon_2),
            ])
            lon_1 += (max(rot_lng)-min(rot_lng))/shape[1]
        lat_1 += (max(rot_lat)-min(rot_lat))/shape[0]
        
    for rot_zone in rot_zones:
        zone = [(coords[0]*math.cos(-φ)-coords[1]*math.sin(-φ), coords[0]*math.sin(-φ)+coords[1]*math.cos(-φ)) for coords in rot_zone]
        zones.append(zone)
        
    print('Done')
    return tuple(zones)


"""
The "requests" function makes the requests using the Google Street View 
API for each subzone and saves the image as well as the subzone in a 
dedicated directory.
"""


def requests(zones, size, heading, pitch, key, dir_img):
    dirs = []
    
    print('Downloading the images...')
    length = len(zones)
    for i, zone in enumerate(zones):
        num = i//(length//20)
        print('['+num*'#'+(20-num)*'.'+f']\t({i}/{length})', end = '\r')

        centroid = (
            ((zone[0][0] + zone[1][0] + zone[2][0] + zone[3][0])/4),
            ((zone[0][1] + zone[1][1] + zone[2][1] + zone[3][1])/4),
        )
        
        location = f'{centroid[0]},{centroid[1]}'
        
        api_args = {
            'size': size,
            'location': location,
            'heading': heading,
            'pitch': pitch,
            'key': key,
        }
        
        dirs.append(location)

        if not os.path.exists(f'{dir_img}/{location}'):
            os.mkdir(f'{dir_img}/{location}')
        
        if not os.path.exists(f'{dir_img}/{location}/metadata.json'):
            api_list = google_streetview.helpers.api_list(api_args)
            results = google_streetview.api.results(api_list)
            results.download_links(f'{dir_img}/{location}')

        if not os.path.exists(f'{dir_img}/{location}/poly.csv'):
            with open(f'{dir_img}/{location}/poly.csv', 'w') as f:
                header = ['Latitude','Longitude']
                writer = csv.writer(f)
                writer.writerow(header)
                for coords in zone:
                    writer.writerow(coords)
                    
    print('['+20*'#'+f']\t({length}/{length})')
    print('Done')
    return dirs    


"""
This function makes use of trained Microsoft Lobe models to determine
the height of the lamps in the images.
"""


def analysis(dir_img, dirs, default_hobs = 9):
    
    print('Loading the models...')
    detect_lamp = ImageModel.load(f'{path}Detect lamp TensorFlow')
    eval_hlamp = ImageModel.load(f'{path}Eval hlamp TensorFlow')
    print('Done')
    
    print('Analyzing the images...')
    length = len(dirs)
    
    for i, img in enumerate(dirs):
        num = i//(length//20)
        print('['+num*'#'+(20-num)*'.'+f']\t({i}/{length})', end = '\r')
        if os.path.exists(f'{dir_img}/{img}/obst.csv'):
            continue
        
        if os.path.exists(f'{dir_img}/{img}/gsv_0.jpg'):
            
            lamp_0 = detect_lamp.predict_from_file(f'{dir_img}/{img}/gsv_0.jpg')
            lamp_1 = detect_lamp.predict_from_file(f'{dir_img}/{img}/gsv_1.jpg')
            
            if eval(lamp_0.prediction) and eval(lamp_1.prediction):
                confidence_0 = lamp_0.labels[0][1]
                confidence_1 = lamp_1.labels[0][1]
                hlamp = eval({True: eval_hlamp.predict_from_file(f'{dir_img}/{img}/gsv_0.jpg').prediction,
                              False: eval_hlamp.predict_from_file(f'{dir_img}/{img}/gsv_1.jpg').prediction}[confidence_0 > confidence_1])
                
            elif eval(lamp_0.prediction):
                hlamp = eval(eval_hlamp.predict_from_file(f'{dir_img}/{img}/gsv_0.jpg').prediction)
                
            elif eval(lamp_1.prediction):
                hlamp = eval(eval_hlamp.predict_from_file(f'{dir_img}/{img}/gsv_1.jpg').prediction)
                
            else:
                hlamp = 0
            
        else:
            hlamp = 0
        
        with open(f'{dir_img}/{img}/obst.csv', 'w') as f:
            header = ['Hlamp','Hobs']
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow([hlamp, default_hobs])
            
    print('['+20*'#'+f']\t({length}/{length})')
    print('Done')

    
"""
The "regroup_data" function collects all data and regroups it in
a single csv file. It also fills in gaps for zones where no GSV image
is available or the model did not recognize any lamps.
"""    


def regroup_data(dir_img, dirs):
    print('Collecting and regrouping data...')
    
    size = len(dirs)
    
    zones_lat = np.zeros((size, 4))
    zones_lng = np.zeros((size, 4))
    
    zones_hlamp = np.zeros((size, 1), dtype = int)
    zones_hobs = np.zeros((size, 1), dtype = int)
    zones_fobs = np.ones((size, 1), dtype = int)
    zones_dobs = np.ones((size, 1), dtype = int)
    
    zones_dobs *= 60
    
    for i, img in enumerate(dirs):
        
        zdata = pd.read_csv(f'{dir_img}/{img}/poly.csv')
        
        zones_lat[i][0] = zdata['Latitude'][0]
        zones_lat[i][1] = zdata['Latitude'][1]
        zones_lat[i][2] = zdata['Latitude'][2]
        zones_lat[i][3] = zdata['Latitude'][3]
        
        zones_lng[i][0] = zdata['Longitude'][0]
        zones_lng[i][1] = zdata['Longitude'][1]
        zones_lng[i][2] = zdata['Longitude'][2]
        zones_lng[i][3] = zdata['Longitude'][3]
        
        
        pdata = pd.read_csv(f'{dir_img}/{img}/obst.csv')
        zones_hlamp[i] = int(pdata.iloc[0]['Hlamp'])
        zones_hobs[i] = int(pdata.iloc[0]['Hobs'])
    
    if False not in (zones_hlamp == 0):
        warnings.warn("No lamps have been detected", NoLampsWarning)
    
    else:        
        print('Done')
        print('Filling in the gaps...')
        
        coords = [tuple(map(float, coord.split(','))) for coord in dirs]
        
        for i, hlamp in enumerate(zones_hlamp):
            if hlamp != 0:
                continue
            distances = np.array([distance(coords[i], coords[j]) for j in range(len(coords))])
            distances[distances == 0] = np.inf
            not_found = True
            while not_found:
                minimum = min(distances)
                j = np.where(distances == minimum)[0][0]
                if zones_hlamp[j] != 0:
                    zones_hlamp[i] = zones_hlamp[j]
                    not_found = False
                else:
                    distances[j] = np.inf
    
    np.savetxt('inventory.csv', 
               np.column_stack(
                   (
                   zones_lat,
                   zones_lng,
                   zones_hlamp,
                   zones_hobs,
                   zones_fobs,
                   zones_dobs,
                   )
               )
               , delimiter = ','
               , header = 'lat1,lat2,lat3,lat4,lng1,lng2,lng3,lng4,hlamp,hobs,fobs,dobs'
               , comments = ''
              )   
    


def main():

    global path
    path = os.path.dirname(__file__).replace('\\','/')+'/'

    with open(f'{path}params.txt', encoding='utf8') as f:
        params = yaml.safe_load(f)
    
    dir_img = params['dir_img']
    shape = params['shape']
    
    assert type(shape) == list, 'Shape must be a list of two positive integers'
    assert len(shape) == 2, 'Shape must be a list of two positive integers'
        
    shape = tuple(shape)
    
    method = params['method']
    
    if method == 'coords':
        polygon = params['coords']
        
        assert type(polygon) == list, 'Polygon must be written as a list of coordinates'
        assert len(polygon) >= 3, 'Polygon must contain at least three coordinates'
            
        polygon = [tuple(map(float, coord.split(','))) for coord in polygon]
        polygon, angle = minimum_bounding_box(polygon)
        
    elif method == 'name':
        region = params['region']
        
        assert type(region) == list, "Region must be a list of format ['city','country']"
        assert len(region) == 2, "Region must be a list of format ['city','country']"
        
        city = region[0]
        country = region[1]
        polygon = OSMrequest(city, country)
        polygon, angle = minimum_bounding_box(polygon)
        polygon_verification(polygon)
        
    else:
        raise ValueError("Please input a valid method in the params.txt file ('name' or 'coords')")
    
    size = '640x640'
    heading = '0;90'
    pitch = '0'
    key = params['dev_key']
    
    shape = autorisation(shape)
    
    zones = polygons(polygon, shape, angle)
    
    dirs = requests(zones, size, heading, pitch, key, dir_img)
    
    analysis(dir_img, dirs)
    
    regroup_data(dir_img, dirs)
    
    os.rename('inventory.csv',f'{dir_img}/inventory.csv')

    

if __name__ == '__main__':
    main()
    print('All done')


