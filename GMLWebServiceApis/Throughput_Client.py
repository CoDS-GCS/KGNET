import sys
import os
kgnet_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
inference_dir = os.path.abspath(os.path.dirname(__file__))

sys.path.append(kgnet_dir)
sys.path.append(inference_dir)

from Constants import *
import requests
from datetime import datetime as dt
from tqdm import tqdm
# Define the API endpoint with query parameters
url = "http://localhost:7770/full_batch_BASELINE/mid/example_model_id"

dblp = {"dataset_name": "DBLP15M_PV_FG", }
file_path_dblp = os.path.join(KGNET_Config.datasets_output_path,'targets','DBLP_D1H1','DBLP_D1H1_1600_FG.csv')

mag = {"dataset_name": "MAG42M_PV_FG", }
file_path_mag = os.path.join(KGNET_Config.datasets_output_path,'targets','MAG_D1H1','MAG_D1H1_1600_FG.csv')

yago = {"dataset_name": "YAGO_FM200"}
file_path_yago = os.path.join(KGNET_Config.datasets_output_path,'targets','YAGO_FM200','YAGO_FM200_1600.csv')

Pipeline1 = {'DBLP200':[dblp, file_path_dblp],
             'DBLP400':[dblp,file_path_dblp],
             'DBLP800':[dblp,file_path_dblp],
            'YAGO200':[yago,file_path_yago],
            'YAGO400':[yago,file_path_yago],
            'YAGO800':[yago,file_path_yago],
            'MAG200':[mag,file_path_mag],
            'MAG400':[mag,file_path_mag],
            'MAG800':[mag,file_path_mag],
            }

Pipeline2 = {'DBLP200':[dblp, file_path_dblp],
             'DBLP400':[dblp,file_path_dblp],
            'YAGO200':[yago,file_path_yago],
            'YAGO400':[yago,file_path_yago],
            'MAG200':[mag,file_path_mag],
            'MAG400':[mag,file_path_mag],
            'DBLP800':[dblp,file_path_dblp],
            'YAGO800':[yago,file_path_yago],
            'MAG400':[mag,file_path_mag],
            }

Pipeline3 = {'DBLP200':[dblp, file_path_dblp],
             'YAGO200':[yago,file_path_yago],
             'MAG200':[mag,file_path_mag],
             'DBLP400':[dblp,file_path_dblp],
            'YAGO400':[yago,file_path_yago],
            'MAG400':[mag,file_path_mag],
            'DBLP800':[dblp,file_path_dblp],
            'YAGO400':[yago,file_path_yago],
            'MAG400':[mag,file_path_mag],}

Pipeline_test = {'DBLP200':[dblp, file_path_dblp],
                # 'YAGO200':[yago,file_path_yago],
                # 'MAG200': [mag, file_path_mag],
                 }

### For testing - manual args ###############################################################
from concurrent.futures import ThreadPoolExecutor,as_completed
def send_request(url, params, file):
    with open(file, "rb") as f:
        files = {"file": ("", f.read())}
    return requests.post(url, params=params, files=files)

# with open(file_path_dblp, "rb") as f:
#     # Include the file in the request
#     files1 = {"file": ("", f.read())}
# with open(file_path_dblp, "rb") as f:
#     # Include the file in the request
#     files2 = {"file": ("", f.read())}
# with ThreadPoolExecutor() as executor:
#     future1 = executor.submit(send_request, url, dblp, files1)
#     # future2 = executor.submit(send_request, url, dblp, files2)
#     print('Sending queries in parallel...')
#     response1 = future1.result()
#     # response2 = future2.result()
# if response1.status_code == 200:
#     print("Response JSON:", response1.json())
# # if response2.status_code == 200:
# #     print("Response JSON:", response2.json())
# import sys
# sys.exit()
##############################################################################################
NUM_CON_API = 3
total_time_start = dt.now()
time_dict = {}
Pipeline = Pipeline_test
with ThreadPoolExecutor(max_workers=NUM_CON_API) as executor:
    print(' -->'.join(Pipeline.keys()))
    futures = {
        executor.submit(send_request, url, data[0], data[1]): name
        for name, data in Pipeline.items()
    }
    for future in tqdm(as_completed(futures),total=len(futures), desc="Processing Requests"):
        request_name = futures[future]
        try:
            result = future.result()  # Blocking call to get the result of the future
            print(f"Result for {request_name}: {result.json()}")
        except Exception as e:
            print(f"Error in {request_name}: {e}")
##############################################################################################
# for dataset,query in tqdm(Pipeline2.items()):
#     time_query_start = dt.now()
#     query_params,file_path = query
#     with open(file_path, "rb") as f:
#         files = {
#             "file": ("", f.read())
#         }
#     try:
#         response = requests.post(url, params=query_params, files=files)
#         if response.status_code == 200:
#             print("Response JSON:", response.json())
#         else:
#             print(f"Failed to make request: {response.status_code}, {response.text}")
#     except Exception as e:
#         print(f"Error occurred: {e}")
#     time_query_end = (dt.now() - time_query_start).total_seconds()
#     time_dict[dataset] = time_query_end
total_time_end = (dt.now() - total_time_start).total_seconds()
print('*'*20)
print(f'NUM_CON_API = {NUM_CON_API}')
print(f'Total pipeline time taken: {total_time_end}')
print('*'*20)

