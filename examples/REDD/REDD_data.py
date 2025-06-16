import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


# 1. Load NetCDF satellite file
def load_satellite_data(file_path):
    ds = xr.open_dataset(file_path)
    df_all = pd.DataFrame(
        {
            "latitude": ds["latitude"].values,
            "longitude": ds["longitude"].values,
            "xco2": ds["xco2"].values,
            "time": ds["time"].values,
        }
    )
    return df_all


# 2. Filter data within China region
def filter_china(df_all):
    return df_all[
        (df_all["latitude"] >= 18)
        & (df_all["latitude"] <= 54)
        & (df_all["longitude"] >= 73)
        & (df_all["longitude"] <= 135)
    ]


# 3. Clean enterprise data
def load_enterprise_data(file_path):
    df = pd.read_excel(file_path)
    df = df.dropna(subset=["纬度", "经度"])
    df = df[np.isfinite(df["纬度"]) & np.isfinite(df["经度"])]
    return df


# 4. Build KDTree for enterprise coordinates
def create_enterprise_tree(df):
    enterprise_coords = df[["纬度", "经度"]].to_numpy()
    return cKDTree(enterprise_coords)


# 5. Perform spatial matching between satellite points and enterprises
def match_satellite_to_enterprise(satellite_df, enterprise_tree, search_radius):
    satellite_coords = satellite_df[["latitude", "longitude"]].to_numpy()
    matches = enterprise_tree.query_ball_point(satellite_coords, r=search_radius)
    return matches


# 6. Construct detailed matching results
def build_match_results(matches, satellite_df, enterprise_df):
    detailed_results = []
    for i, matched_indices in enumerate(matches):
        if matched_indices:
            sat_row = satellite_df.iloc[i]
            for idx in matched_indices:
                ent_row = enterprise_df.iloc[idx]
                detailed_results.append(
                    {
                        "Satellite Latitude": sat_row["latitude"],
                        "Satellite Longitude": sat_row["longitude"],
                        "Satellite CO₂ Concentration (xco2)": sat_row["xco2"],
                        "Satellite Observation Time": sat_row["time"],
                        "Province": ent_row.get("省份", None),
                        "City": ent_row.get("城市", None),
                        "Enterprise Name": ent_row.get("企业名称", None),
                        "Emission Point Name": ent_row.get("排放口名称", None),
                        "Enterprise Longitude": ent_row.get("经度", None),
                        "Enterprise Latitude": ent_row.get("纬度", None),
                        "Enterprise Time": ent_row.get("时间", None),
                        "Enterprise CO₂ Emission (kg)": ent_row.get(
                            "二氧化碳排放量(kg)", None
                        ),
                    }
                )
    return pd.DataFrame(detailed_results)


# 7. Aggregate data (remove duplicates and merge)
def aggregate_data(df):
    group_keys = [
        "Enterprise Name",
        "Match Time",
        "Satellite Latitude",
        "Satellite Longitude",
    ]
    aggregated_df = df.groupby(group_keys, as_index=False).agg(
        {
            "Enterprise CO₂ Emission (kg)": "sum",
            "Satellite CO₂ Concentration (xco2)": "first",
            "Satellite Observation Time": "first",
            "Province": "first",
            "City": "first",
            "Enterprise Longitude": "first",
            "Enterprise Latitude": "first",
            "Enterprise Time": "first",
        }
    )
    return aggregated_df


# 8. Remove duplicate rows
def deduplicate_data(df):
    return df.drop_duplicates(subset=["Enterprise Name", "Match Time"], keep="first")


# 9. Load weather data
def load_weather_data(file_path):
    return pd.read_csv(file_path)


# 10. Preprocess weather data
def preprocess_weather_data(df_meteo):
    df_meteo["date"] = pd.to_datetime(df_meteo["date"])
    df_meteo["Standardized City"] = df_meteo["city"].str.replace("市", "", regex=False)
    return df_meteo


# 11. Match weather data by time and city
def smart_city_match(row, meteo):
    time = row["Match Time"]
    city = row["Standardized City"]

    if city in ["辖区", "市辖区", "县"]:
        city = row["Province"].replace("省", "").replace("市", "")

    candidates = meteo[
        (meteo["Standardized City"] == city)
        & (meteo["date"] >= time - pd.Timedelta(hours=2))
        & (meteo["date"] <= time + pd.Timedelta(hours=2))
    ]

    if not candidates.empty:
        return candidates.sort_values("date").iloc[0][
            ["prs", "winDAvg2mi", "winSAvg2mi", "tem", "rhu", "pre3h"]
        ]
    else:
        return pd.Series(
            [None] * 6, index=["prs", "winDAvg2mi", "winSAvg2mi", "tem", "rhu", "pre3h"]
        )


# 12. Merge weather data with enterprise-satellite matched data
def merge_weather_data(df_enterprise, df_meteo):
    matched_weather = df_enterprise.apply(
        lambda row: smart_city_match(row, df_meteo), axis=1
    )
    return pd.concat([df_enterprise, matched_weather], axis=1)


# === Main Process ===
satellite_file_path = (
    "https://disc.gsfc.nasa.gov/datasets?keywords=oco2&page=1"  # nasa data
)
enterprise_file_path = "./20240101_data.xlsx"
weather_file_path = "./20240101_20240301_meteo_data.csv"

# Load and filter satellite data
satellite_df = load_satellite_data(satellite_file_path)
satellite_df_china = filter_china(satellite_df)

# Load enterprise data
enterprise_df = load_enterprise_data(enterprise_file_path)

# Build KDTree for spatial search
enterprise_tree = create_enterprise_tree(enterprise_df)

# Define matching radius (approx. pixel size)
lat_half = 2.25 / 111 / 2
lon_half = 1.29 / 111 / 2
search_radius = max(lat_half, lon_half)

# Match satellite pixels to nearby enterprises
matches = match_satellite_to_enterprise(
    satellite_df_china, enterprise_tree, search_radius
)

# Build detailed match results
detailed_df = build_match_results(matches, satellite_df_china, enterprise_df)

# Aggregate matched data
aggregated_df = aggregate_data(detailed_df)

# Deduplicate entries
deduplicated_df = deduplicate_data(aggregated_df)

# Load and process meteorological data
df_meteo = load_weather_data(weather_file_path)
df_meteo = preprocess_weather_data(df_meteo)

# Merge meteorological data with matched results
final_df = merge_weather_data(deduplicated_df, df_meteo)

# Save final merged dataset
final_df.to_excel("merged_result.xlsx", index=False)
