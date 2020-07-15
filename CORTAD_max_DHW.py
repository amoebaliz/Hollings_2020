import numpy as np
import netCDF4 as nc
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as pltd
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import shapefile as shp

# OBJECTIVE: Map of the coral triangle with maximum DHW during 1998
# IDEAS: 1) Alternate Calculation of DHW
#        2) Inset map of the Philippines
#        3) Convert time to dates for the 'WHEN' plot

def convert_time(time):
    # NOTE: TIME VARIABLES MUST BE IN DAYS SINCES (1900,1,1,0,0)
    ref = dt.datetime(1900,1,1,0,0)
    date_vals = np.zeros(len(time))
    for nt in range(len(time)):
        day_time = ref + dt.timedelta(days=np.float(time[nt]))
        date_vals[nt] = pltd.date2num(day_time)
    return date_vals

def polygon_patch(mapid,axs):
    mapid.drawcoastlines(linewidth=0)
    polys = []
    for polygon in mapid.landpolygons:
        polys.append(polygon.get_coords())

    lc = PolyCollection(polys, edgecolor='black',
         facecolor='0.8', closed=False)
    axs.add_collection(lc)

def plt_shp_coord(sf,m_fin):
    m0 = Basemap(projection = 'cea')
    x_0,y_0 = m0(0,0)
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        x = np.array(x)-x_0
        y = np.array(y)+y_0
        lons,lats = m0(x,y,inverse=True)
        m_fin.plot(lons,lats,'.1',lw=6)
        m_fin.plot(lons,lats,'paleturquoise',lw=1.5)

def plt_box(ax):

    ax.plot((lons[j0,i0],lons[j0,iN]),(lats[j0,i0],lats[j0,iN]),'-w',lw=1.5)
    ax.plot((lons[j0,i0],lons[jN,i0]),(lats[j0,i0],lats[jN,i0]),'-w',lw=1.5)
    ax.plot((lons[j0,iN],lons[jN,iN]),(lats[j0,iN],lats[jN,iN]),'-w',lw=1.5)
    ax.plot((lons[jN,i0],lons[jN,iN]),(lats[jN,i0],lats[jN,iN]),'-w',lw=1.5)

##############################################

# FILE DETAILS
dhw_file = '/Volumes/P1/Data/CORTAD/Version4/cortadv4_TSA_DHW_coral_1998-1999.nc'
fid = nc.Dataset(dhw_file)

cmap_file = '/Volumes/P4/workdir/liz/ANALYSES/VIP/paper_figs/dhw_noaa.cmap'

shp_file = '/Volumes/P4/workdir/liz/external_data/CT_ATLAS/Coral_Triangle_Boundary/Coral_Triangle_Boundary_Line.shp'
sfid = shp.Reader(shp_file)

# EXTRACT VARIABLES
time = fid.variables['time'][:60]
plot_time = convert_time(time)
lats = fid.variables['lat_bounds'][:]
lons = fid.variables['lon_bounds'][:]

DHW = fid.variables['dhw'][:60,:,:]
DHW = np.ma.masked_where(DHW<0,DHW)
DHW_max = np.max(DHW,axis=0)
DHW_max_when = np.argmax(DHW,axis=0)

cmap = np.loadtxt(cmap_file)
lev_seven = np.array((159,212,0))
cmap2 = np.vstack((cmap[0:7,:],lev_seven,cmap[7:,:]))
cmap = cmap2/256.
dhw_noaa = colors.ListedColormap(cmap)

# REGION BOUNDS
j0 = [860]
jN = [875]
i0 = [735]
iN = [749]

# FIGURE DETAILS
fig = plt.figure(figsize=(15,9))
ax1 = fig.add_subplot(111)

cmin=0
cmax=16
# main map domain
lats1 = [-22.,25.]
lons1 = [90.,170.]
# inset map domain
lats2 = [12.4,15]
lons2 = [120.,122.5]

# PRIMARY CT MAP
m = Basemap(llcrnrlon=lons1[0],llcrnrlat=lats1[0],urcrnrlon=lons1[1],urcrnrlat=lats1[1],resolution='i')
m.pcolormesh(lons,lats,DHW_max,vmin=cmin,vmax=cmax,cmap=dhw_noaa,latlon=True)
polygon_patch(m,ax1)
plt_shp_coord(sfid,m)

m.drawmeridians([100,120,140,160], labels=[0,0,1,0], fmt='%d', fontsize=18)
m.drawparallels([-20,0,20], labels=[1,0,0,0], fmt='%d', fontsize=18)

#cb_ticks = np.linspace(cmin,cmax,5)
#clb = m.colorbar(ticks=cb_ticks)
#for t in clb.ax.get_yticklabels():
#    t.set_fontsize(18)

cb_ticks = np.arange(17)
clb = m.colorbar(ticks=[],location='bottom',size="6%", pad="3%")
for t in cb_ticks:
    clb.ax.text((2.0*t+1)/(2*17),0.5,str(int(t)),fontsize = 18, va='center',ha = 'center')

tpos = np.array([5.0,12.0,24.0])/34
tval = ['Possible', 'Likely', 'Mortality Likely']
col = [2,4,12]
apos = np.array([2.0,8.0,16.0])/34 +.01
bpos = np.array([8.0,16.0,34.0])/34 -.01

for nt in range(3):
    if nt < 2:
       clb.ax.plot([apos[nt],bpos[nt]],-.4*np.ones(2),lw = 2, clip_on=False, \
                color = dhw_noaa.colors[col[nt],:])
    else:
       clb.ax.annotate("", xy=(bpos[nt]+.01, -.4), xycoords='data', xytext=(apos[nt],-.4), textcoords='data', \
            arrowprops= dict(lw = 2, arrowstyle="->,head_length=.8, head_width=.4", color=dhw_noaa.colors[col[nt],:]),\
            annotation_clip=False)

    clb.ax.text(tpos[nt],-1,tval[nt],fontsize = 18, va='center',ha = 'center',clip_on=False)

# INSET VIP MAP
axins = zoomed_inset_axes(ax1, 9.2, loc=1, borderpad=.06)

axins.set_xlim(lons2[0], lons2[1])
axins.set_ylim(lats2[0], lats2[1])

m2 = Basemap(llcrnrlon=lons2[0],llcrnrlat=lats2[0],urcrnrlon=lons2[1],urcrnrlat=lats2[1],resolution='f')
m2.drawmapboundary(fill_color='azure',linewidth=2.0)
m2.pcolormesh(lons,lats,DHW_max,vmin=0,vmax=16,cmap=dhw_noaa,latlon=True)
polygon_patch(m2,axins)

axins.plot(121,14.5,'D',ms=8,mfc=dhw_noaa.colors[9,:],mec='k')
axins.text(121.09,14.5,'Manila',style='italic', fontsize = 11, va='center')
axins.text(121.08,13.1,'Mindoro',fontsize = 14, ha='center', va='center')
axins.text(121.2,14.08,'Luzon',fontsize = 14, ha='center', va='center')

mark_inset(ax1, axins, loc1=2, loc2=3, fc="none", ec="0",lw=2)
plt_box(axins)

plt.savefig('Fig1_Max_DHW_98.png')
plt.show()
