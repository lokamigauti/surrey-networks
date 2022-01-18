import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from geocube.api.core import make_geocube

DATA_DIR = 'G:/My Drive/IC/Doutorado/Sandwich/Data/'
ROADS = 'Roads/'
FUNCTIONS = 'Functions/'
WOKING_OUTSKIRTS_LAT_MIN = 51.2844718
WOKING_OUTSKIRTS_LAT_MAX = 51.3494766
WOKING_OUTSKIRTS_LON_MIN = -0.6344602
WOKING_OUTSKIRTS_LON_MAX = -0.4544372
RESOLUTION = (100, 100)

if __name__ == '__main__':
    woking_outskirts = Polygon([(WOKING_OUTSKIRTS_LON_MIN, WOKING_OUTSKIRTS_LAT_MIN),
                                (WOKING_OUTSKIRTS_LON_MIN, WOKING_OUTSKIRTS_LAT_MAX),
                                (WOKING_OUTSKIRTS_LON_MAX, WOKING_OUTSKIRTS_LAT_MAX),
                                (WOKING_OUTSKIRTS_LON_MAX, WOKING_OUTSKIRTS_LAT_MIN),
                                (WOKING_OUTSKIRTS_LON_MIN, WOKING_OUTSKIRTS_LAT_MIN)])
    roads_SU = gpd.read_file(DATA_DIR + ROADS + 'SU_RoadLink.shp')
    roads_TQ = gpd.read_file(DATA_DIR + ROADS + 'TQ_RoadLink.shp')

    woking_outskirts_gdf = gpd.GeoDataFrame([1], geometry=[woking_outskirts], crs='EPSG:4326').to_crs('EPSG:27700')
    roads_SU_outskirts = roads_SU.clip(woking_outskirts_gdf)
    roads_TQ_outskirts = roads_TQ.clip(woking_outskirts_gdf)
    roads = gpd.GeoDataFrame(pd.concat([roads_SU_outskirts, roads_TQ_outskirts], ignore_index=True))

    # roads.drop(columns=roads.columns.difference(['function']))

    roads.to_file(DATA_DIR + ROADS + 'roads.shp')

    categorical_enums = {'function': roads['function'].unique()}

    out_grid = make_geocube(
        vector_data=roads,
        resolution=RESOLUTION,
        categorical_enums=categorical_enums
    )

    out_grid.function.plot()
    plt.show()

    function_string = out_grid['function_categories'][out_grid['function'].astype(int)] \
        .drop('function_categories')

    out_grid['function'] = function_string
    pdf = out_grid.drop(['spatial_ref', 'function_categories']).to_dataframe()
    cat_dtype = pd.api.types.CategoricalDtype(out_grid.function_categories.values)
    pdf['function'] = pdf['function'].astype(cat_dtype)

    training_df = pd.get_dummies(pdf, columns=['function'])
    training_df.head()


    # for road_func in roads['function'].unique():
    #     roads_subset = roads.loc[roads['function'] == road_func]
    #     out_grid = make_geocube(
    #         vector_data=roads_subset,
    #         resolution=RESOLUTION,
    #     )
    #
    #     file_name = road_func + '_' + str(RESOLUTION[0]) + 'x' + str(RESOLUTION[1]) + '.tif'
    #     out_grid.rio.to_raster(DATA_DIR + ROADS + FUNCTIONS + file_name.replace(' ', '_'))
