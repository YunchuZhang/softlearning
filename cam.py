num_angles = 18
num_elevs = 3
start_angle = 0
angle_delta= 10
start_elevation = -120 
elevation_delta = -20
elev_ang = []
for angle_i in range(num_angles):
    for elev_i in range(num_elevs): 
    	elev_ang.append((start_elevation + elevation_delta*elev_i,start_angle + angle_delta*angle_i))
print(elev_ang)