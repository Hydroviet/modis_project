import os
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import json
from datetime import datetime

class GetModisData:
    """A class for easily download a list of subsets of MODIS data.

    You only have to learn how to use the following method:
        GetModisData.create_order_url(reservoirs_GeoDataFrame,
                                      modis_product_code,
                                      recreate_download_file,
                                      download_result_file,
                                      start_index, n_downloaded_reservoirs,
                                      email_id)

    Example:
        reservoir_GeoDataFrame = gpd.read_file('mekongReservoirs')
        GetModisData.create_order_url(reservoir_GeoDataFrame,
                                      modis_product_code='MOD13Q1',
                                      recreate_download_file=True,
                                      download_result_file='MOD13Q1_a',
                                      start_index=0, n_downloaded_reservoirs=1,
                                      email_id=1)
    """

    URL = 'https://modis.ornl.gov/rst/api/v1/'
    HEADER = {'Accept': 'application/json'} 
    SUBSET_URL = 'https://modis.ornl.gov/subsetdata/'
    
    def cal_distance_in_km(point1, point2):
        import pyproj
        geod = pyproj.Geod(ellps='WGS84')
        angle1, angle2, distance = geod.inv(point1.x, point1.y, 
                                            point2.x, point2.y)
        return distance/1000

    
    def find_rect_bound(polygon):
        from shapely.geometry import Point

        xmin, ymin, xmax, ymax = polygon.bounds
        top_left = Point(xmin, ymax)
        bottom_left = Point(xmin, ymin)
        top_right = Point(xmax, ymax)
        bottom_right = Point(xmax, ymin)
        vertices = [top_left, top_right, bottom_left, bottom_right]

        centroid = polygon.centroid
        km_above_below = km_left_right \
                       = np.int32(np.ceil(np.max([
                           GetModisData.cal_distance_in_km(centroid, v) 
                           for v in vertices])))
        return centroid, km_above_below, km_left_right
    
    
    def get_modis_date(reservoir_row_df):
        # Submit request
        row = reservoir_row_df
        response = requests.get(
            ''.join(
                [GetModisData.URL, row['product'],
                 '/dates?latitude=', str(row['latitude']),
                 '&longitude=', str(row['longitude'])]
            ),
            headers=GetModisData.HEADER
        )

        # Get dates object as list of python dictionaries
        dates = json.loads(response.text)['dates']

        # Convert to list of tuples;
        # change calendar_date key values to datetimes
        dates = [(datetime.strptime(date['calendar_date'], "%Y-%m-%d"),
                  date['modis_date']) for date in dates]

        # Get MODIS dates nearest to start_date and end_date
        # and add to new pandas columns
        start_MODIS_date = min(date[1] for date in dates 
                if date[0] > row['start_date'])
        end_MODIS_date = max(date[1] for date in dates 
                if date[0] < row['end_date'])

        return start_MODIS_date, end_MODIS_date
    
    
    def create_reservoirs_download_dataframe(reservoirs_GeoDataFrame,
                                             product_code,
                                             email,
                                             start_index,
                                             start_date,
                                             end_date):
        df = pd.DataFrame(columns=['site_id', 'product', 
                                   'latitude', 'longitude', 
                                   'email', 'start_date', 'end_date',
                                   'km_above_below', 'km_left_right', 
                                   'order_uid'])
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        n_downloaded = 0
        
        for index, reservoir_row in reservoirs_GeoDataFrame.iterrows():
            center, km_above_below, km_left_right = \
                GetModisData.find_rect_bound(reservoir_row.geometry)
            row = {'site_id': 'site{}'.format(index),
                   'product': product_code,
                   'latitude': center.y,
                   'longitude': center.x,
                   'email': email,
                   'start_date': start_date,
                   'end_date': end_date,
                   'km_above_below': km_above_below,
                   'km_left_right': km_left_right,
                   'order_uid': 'None'}
            
            try:
                row['start_date'], row['end_date'] = \
                    GetModisData.get_modis_date(row)
                df = df.append(row, ignore_index=True)
                n_downloaded += 1
            except:
                print('Invalid argument: Number of kilometers above or below \
                       the subset location must be less than or equal to 100.')
        
        df.index = pd.RangeIndex(start_index, start_index + n_downloaded)
        return df
    
    
    def _create_order_url(reservoirs_download_df, download_result_file=None):
        # Make list to collect order UIDs
        order_uids = []
        n_downloaded = 0
        df = pd.DataFrame(columns=['site_id', 'product', 
                                   'latitude', 'longitude', 
                                   'email', 'start_date', 'end_date',
                                   'km_above_below', 'km_left_right', 
                                   'order_uid'])

        for index, row in reservoirs_download_df.iterrows():
            if row['order_uid'] == 'None':
                # Build request URL
                request_url = ''.join(
                    [GetModisData.URL, row['product'],
                     '/subsetOrder?latitude=', str(row['latitude']),
                     '&longitude=', str(row['longitude']),
                     '&email=', row['email'],
                     '&uid=', row['site_id'],
                     '&startDate=', row['start_date'],
                     '&endDate=', row['end_date'],
                     '&kmAboveBelow=', str(row['km_above_below']),
                     '&kmLeftRight=', str(row['km_left_right'])]
                )

                print(request_url)
                # Submit request
                response = requests.get(
                    request_url, 
                    headers=GetModisData.HEADER)
                try:
                    order_uid = (GetModisData.SUBSET_URL
                                 + json.loads(response.text)['order_id'])
                    row['order_uid'] = order_uid
                    n_downloaded += 1
                    df = df.append(row, ignore_index=True)
                    order_uids.append(order_uid)
                except TypeError as err:
                    print(response.text)

        df.index = pd.RangeIndex(0, n_downloaded)
        print(order_uids)
        if download_result_file is None:
            df.to_csv(product_code + '.csv')
        else:
            df.to_csv(download_result_file + '.csv')
        return order_uids
    
    
    # Main function to get MODIS data
    def create_order_url(reservoirs_GeoDataFrame, 
                         modis_product_code='MOD13Q1', 
                         recreate_download_file=False,
                         download_result_file=None, 
                         start_index=0, n_downloaded_reservoirs=1,
                         email_id=3,
                         start_date='2000-01-01',
                         end_date='2018-12-31'):
        """Create order url on MODIS web service host.

        Example:
            reservoir_GeoDataFrame = gpd.read_file('mekongReservoirs')
            GetModisData.create_order_url(reservoir_GeoDataFrame,
                                          modis_product_code='MOD13Q1',
                                          recreate_download_file=True,
                                          download_result_file='MOD13Q1_a',
                                          start_index=0,
                                          n_downloaded_reservoirs=1,
                                          email_id=1,
                                          start_date='2000-01-01',
                                          end_date='2018-12-31')

        Args:
            reservoirs_GeoDataFrame: A geopandas dataframe, read from a .shp file.
            modis_product_code: String code of MODIS product.
            recreate_download_file: Boolean, indicating whether recreating 
                a new result file or not
            download_result_file: Path of donwload file containing url to 
                download data.
            start_index: Index of first downloaded reservoir to download.
            n_downloaded_reservoirs: Number of downloaded reservoirs counted from
                start_index.
            email_id: Id of the first usable email in this day (one email can request 
                21 times per day max). You should carefully check this argument to
                prevent error.
            start_date: First date to collect data, a string "YYYY-MM-DD" format.
            end_date: Last date to collect data, a string "YYYY-MM-DD" format.
        
        Returns:
            List of order uids.
        """
        try:
            os.makedirs('DownloadFiles')
        except:
            pass

        if download_result_file is None:
            download_result_file = modis_product_code
        reservoirs_download_df = pd.DataFrame(columns=['site_id', 'product',
                                                       'latitude', 'longitude', 
                                                       'email', 'order_uid',
                                                       'start_date', 'end_date',
                                                       'km_above_below',
                                                       'km_left_right'])
        if recreate_download_file:
            write_mode = 'w'
        else:
            write_mode = 'a'
        end_index = start_index + n_downloaded_reservoirs

        if not os.path.exists(download_result_file + '.csv') or recreate_download_file:
            with open(download_result_file + '.csv', write_mode) as f:
                if email_id is None:
                     email_id = 3
                if n_downloaded_reservoirs >= 20:
                    for idx in np.arange(start_index, end_index - end_index % 20, 20):
                        email = '1616' + str(email_id).zfill(3) + '@student.hcmus.edu.vn'
                        download_GeoDataFrame = reservoirs_GeoDataFrame.iloc[idx:idx + 20]
                        df = GetModisData.\
                            create_reservoirs_download_dataframe(download_GeoDataFrame,
                                                                 modis_product_code, 
                                                                 email, idx,
                                                                 start_date,
                                                                 end_date) 
                        reservoirs_download_df = reservoirs_download_df.append(df,
                                                                ignore_index=True)
                        email_id += 1
                
                email = '1616' + str(email_id).zfill(3) + '@student.hcmus.edu.vn'
                download_GeoDataFrame = reservoirs_GeoDataFrame.\
                                    iloc[end_index - end_index % 20 : end_index]
                df = GetModisData.\
                    create_reservoirs_download_dataframe(download_GeoDataFrame,
                                                         modis_product_code, 
                                                         email, 
                                                         end_index - end_index % 20,
                                                         start_date,
                                                         end_date)
                reservoirs_download_df = reservoirs_download_df.append(df,
                                                            ignore_index=True)
                reservoirs_download_df.to_csv(f, header=(start_index==0))
        else:
            reservoirs_download_df = pd.read_csv(download_result_file + '.csv')
            # Append until enough order_uid
            # Not implemented yet
        return GetModisData._create_order_url(reservoirs_download_df, 
                                              download_result_file)
