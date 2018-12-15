
""" Quality and fast triangular mesh generation using Jonathan Shewchuk's triangle algorithm (jrs@cs.berkeley.edu)
    modified to allow input of boundary shapefile and input lines which will be respected by the triangulation
    Author: Antony Orton  26 August 2018  """



import triangle as tri
import triangle.plot
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.path as matpath
import shapely.geometry as shp
from shapely.geometry import mapping
from shapely.ops import cascaded_union
import shapely.affinity as aff
import descartes as dc
import fiona
import os
from time import time



def triMeshgen(bounding_poly,lines = [],points = [],polygons = [],target_length = 5.0,element_area = 20000,buffer_dist = 0):

    """
    Quality triangular mesh generation using Jonathan Shewchuk's triangle algorithm
    Minimum angle of any triangle can be specified (28 degrees seems to work, see end of function)
    
    INPUTS:
    target_length (float): target point (node) spacing along input lines / polygons
    element_area (float): Maximum allowed element area
    bounding_poly (shapely polygon): boundary of the triangulation
    lines (list of shapely linestrings): Input lines which will be preserved by the triangulation
    polygons (list of shapely polygons): Input polygons which will be preserved  by the triangulation
    points (list of shapely points): Input points which will be preserved  by the triangulation
    buffer_dist (float): buffer distance around lines for which target element size will decreased. Set as 0 if no buffer desired
    """
    
    if len(polygons)>0:
        print('NOTE: input of interior "polygons" not yet implemented')
        print('Please convert to polygons to lines (ie shapely LineStrings) and re-try')

        
        
    print('grading lines to target node spacing ..')
    #get rid of multilinestrings and short lines
    lines_new=[]
    for i in range(len(lines)):
        if type(lines[i])==shp.LineString:
            if lines[i].length>1.5*target_length:
                lines_new.append(lines[i])
        else:
            for j in range(len(lines[i])):
                if lines[i][j].length>1.5*target_length:
                    lines_new.append(lines[i][j])
    lines = lines_new
    

    #get rid of multipolygons
    polygons_new=[]
    for i in range(len(polygons)):
        if type(polygons[i])==shp.Polygon:
                polygons_new.append(polygons[i])
        else:
            for j in range(len(polygons[i])):
                    polygons_new.append(polygons[i][j])
    polygons = polygons_new
    
    
    
    #grade the lines / input polygons to target_length distance between nodes along lines. Convert and append polygons to lines
    for i in range(len(lines)):
        lines[i] = grader(lines[i], type = 'line', target_length = target_length)   
    for i in range(len(polygons)):
        poly1 = grader(polygons[i].boundary, type = 'line', target_length = target_length)
        lines.append(poly1)
 
    
    print('buffering lines for refinement near lines ...')
    #buffer all lines and get representative points for refinement inside buffers
    a_inner = target_length**2 
    area_in_buffer = 5*a_inner  #desired element areas in side buffer  (trial and error)

    rep_points=[]
    if buffer_dist>0:
        bufs=[lines[i].buffer(buffer_dist) for i in range(len(lines))]
        buf_union=cascaded_union(bufs)
        buf_union=buf_union.simplify(tolerance = target_length/2)
        if type(buf_union)==shp.Polygon:
            rep_points.append(np.hstack((np.array(buf_union.representative_point().coords[0]),np.array([0,area_in_buffer]))))
        else:
            for i in range(len(buf_union)):
                #NOTE: Must specify as: [point xy inside the polygon to be refined ,0, area] where area is desired element area inside polygon
                rep_points.append(np.hstack((np.array(buf_union[1].representative_point().coords[0]),np.array([0,area_in_buffer]))))
         
    

    #Add buffer boundaries to buf_lines - must deal with annoying cases of multiline and multipolygons
    if buffer_dist>0:
        buf_lines = []
        if type(buf_union)==shp.Polygon:
            if type(buf_union.boundary)==shp.LineString:
                buf_lines.append(buf_union.boundary)
            else:
                for j in range(len(buf_union.boundary)):
                    buf_lines.append(buf_union.boundary[j])
        else:
            for i in range(len(buf_union)):
                if type(buf_union[i].boundary)==shp.LineString:
                    buf_lines.append(buf_union[i].boundary)
                else:
                    for j in range(len(buf_union[i].boundary)):
                        buf_lines.append(buf_union[i].boundary[j])
               
    
    print('preparing triangle input data ..')
    tri_data={}

    # input the boundary  - holes not done yet (ie they are ignored)
    if type(bounding_poly.boundary)==shp.LineString:
        a=np.array(bounding_poly.boundary.coords[0:-1])   #edge vertices 
    else:    
        a=np.array(bounding_poly.boundary[0].coords[0:-1])   #edge vertices 
    b=np.vstack((np.arange(0,len(a)),np.arange(1,len(a)+1))).T    #connection ordering of edges 
    b[-1][1]=0
    tri_data['vertices']=a
    tri_data['segments']=b   
             
    
    #Add input lines
    for i in range(len(lines)):
        a=np.array(lines[i].coords)  
        a=a[:,0:2] #exclude z coordinates
        if buffer_dist == 0:  #only added as segments if no buffer distance specified as this interrups the buffer area refinement
            b=np.vstack((np.arange(0,len(a)),np.arange(1,len(a)+1))).T    #connection ordering of edges 
            b=b[0:-1] # not a closed loop so delete last connection
            b+=len(tri_data['vertices'])
            tri_data['segments']=np.vstack((tri_data['segments'],b))
        tri_data['vertices']=np.vstack((tri_data['vertices'],a))
    
    #Add buffer lines
    if buffer_dist>0:
        for i in range(len(buf_lines)):
            a=np.array(buf_lines[i].coords)  
            a=a[:,0:2] #exclude z coordinates
            b=np.vstack((np.arange(0,len(a)),np.arange(1,len(a)+1))).T    #connection ordering of edges 
            b=b[0:-1] # not a closed loop so delete last connection
            b+=len(tri_data['vertices'])
            tri_data['segments']=np.vstack((tri_data['segments'],b))
            tri_data['vertices']=np.vstack((tri_data['vertices'],a))

        
    #add polygons  (not yet done)
    ####
  
    
    #add regions with rep points for refinement near lines  
    if buffer_dist>0:
        tri_data['regions']=np.array(rep_points)[0]
    
    
    print('tiangulating ..')
    t1=time()
    
    super_triangle=tri.triangulate(tri_data,opts='piq28a'+str(element_area)+'a')  #main triangulation  -  must use incremental algorithm (i switch)
    #Algorithm is copyright Jonathan Shewchuck jrs@cs.berkeley.edu  (respect)
    
    t1=str(time()-t1)
    print(str(len(super_triangle['vertices']))+' vertices')
    print(str(len(super_triangle['triangles']))+' triangles')
    print('time taken: '+t1[0:6]+' seconds')
    
    return super_triangle
       
def grader(shapein, type = 'line', target_length = 2.0):

    """ makes sure line coords are equally spaced points of dist equal to target_length"""

    if type == 'line':    #make line consist of equally spaced points
        line = shapein
        coords=[line.interpolate(i*target_length).coords[0] for i in range(int(line.length/target_length))]
        coords.append(line.coords[-1])
        line=shp.LineString(coords)
    else:
        print('Input type is not a line. Exit function.')
        return
        
    return line

def write_tri_to_shapefile(triangulation,output_point_shapefile = 'SuperTri_points.shp',output_poly_shapefile = 'SuperTri_poly.shp'):

    """ Writes a triangulation to two ESRI shapefiles. 
        Triangulation: Dictionary containing keys: 'vertices' (point coordinates array) and 'triangles' (point connectivity array)
        """
    
    super_tri = triangulation
    
    print('creating shapely point list ..')
    points = [shp.Point(super_tri['vertices'][i]) for i in range(len(super_tri['vertices']))]
    print('creating shapely polygon list ..')
    polygons = [shp.Polygon(super_tri['vertices'][super_tri['triangles'][i]]) for i in range(len(super_tri['triangles']))]


    print('writing point shapefile ..')
    t1=time()

    # Write a new Shapefile - points
    schema = {
    'geometry': 'Point',
    'properties': {'id': 'int'},
    }
        
    with fiona.open(output_point_shapefile, 'w', 'ESRI Shapefile', schema) as c:
        for i in range(len(points)):
            c.write({
                'geometry': mapping(points[i]),
                'properties': {'id': i,},
            })
    print(str(time()-t1)[0:8]+' seconds')       


    print('writing polygon shapefile ..')
    t1=time()

    # Write a new Shapefile - polygons
    schema = {
    'geometry': 'Polygon',
    'properties': {'id': 'int','vertices':'str'},
    }

    with fiona.open(output_poly_shapefile, 'w', 'ESRI Shapefile', schema) as c:
        for i in range(len(polygons)):
            c.write({
                'geometry': mapping(polygons[i]),
                'properties': {'id': i,'vertices':str(super_tri['triangles'][i]).strip('[').strip(']')},
            })
    print(str(time()-t1)[0:8]+' seconds')
    
    return 'patience is a virtue'

    
if __name__  ==  "__main__":

    #Input data - test
    #poly=shp.Polygon(((0,0),(1000,0),(1000,800),(0,800)))
    #line1=shp.LineString(np.array([[200,200],[900,600]]))
    #line2=shp.LineString(np.array([[250,600],[350,100]]))
    #lines=[line1,line2]
    #target_length = 5
    #element_area = 5000
    #buffer_dist=25
    
    
    ###Input data
    nsw = gpd.read_file('NSWborder.shp')
    poly = np.array(nsw.geometry)[0].simplify(tolerance = 25)
    rail = gpd.read_file('NSWallrail.shp')
    lines = [rail.iloc[i].geometry for i in range(len(rail))]
    target_length = 250
    element_area = 5000000000
    buffer_dist=1600
    
    
    super_triangle = triMeshgen(bounding_poly = poly, lines = lines,target_length = target_length,element_area = element_area,buffer_dist=buffer_dist)
    

       
    #plotting
    fig,ax=plt.subplots()
    rail.plot(ax=ax,color='r',linewidth=1)
    triangle.plot.plot(ax,vertices=super_triangle['vertices'],triangles=super_triangle['triangles'])
    
    plt.axis('equal')
    plt.show()