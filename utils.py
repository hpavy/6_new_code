#### Les fonctions utiles ici
import pandas as pd
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from model import PINNs
import torch
import scipy

def write_csv(data, path, file_name):
    dossier = Path(path)
    df = pd.DataFrame(data)
    # Créer le dossier si il n'existe pas
    dossier.mkdir(parents=True, exist_ok=True)
    df.to_csv(path + file_name)


def read_csv(path):
    return pd.read_csv(path)


def charge_data():
    """
    Charge the data of X_full, U_full with every points
    And X_train, U_train with less points
    """
    # # La data
    # # On adimensionne la data
    # df = pd.read_csv("data.csv")
    # df_modified = df[
    #     (df["Points:0"] >= 0.015)
    #     & (df["Points:0"] <= 0.2)
    #     & (df["Points:1"] >= -0.1)
    #     & (df["Points:1"] <= 0.1)
    #     & (df["Time"] > 4)
    #     & (df["Time"] < 6)
    # ]
    # # Uniquement la fin de la turbulence

    # x_full, y_full, t_full = (
    #     np.array(df_modified["Points:0"]),
    #     np.array(df_modified["Points:1"]),
    #     np.array(df_modified["Time"]),
    # )
    # u_full, v_full, p_full = (
    #     np.array(df_modified["Velocity:0"]),
    #     np.array(df_modified["Velocity:1"]),
    #     np.array(df_modified["Pressure"]),
    # )

    # x_norm_full = (x_full - x_full.mean()) / x_full.std()
    # y_norm_full = (y_full - y_full.mean()) / y_full.std()
    # t_norm_full = (t_full - t_full.mean()) / t_full.std()
    # p_norm_full = (p_full - p_full.mean()) / p_full.std()
    # u_norm_full = (u_full - u_full.mean()) / u_full.std()
    # v_norm_full = (v_full - v_full.mean()) / v_full.std()

    # X_full = np.array([x_norm_full, y_norm_full, t_norm_full], dtype=np.float32).T
    # U_full = np.array([u_norm_full, v_norm_full, p_norm_full], dtype=np.float32).T

    # x_int = np.linspace(x_norm_full.min(), x_norm_full.max(), 8)
    # y_int = np.linspace(y_norm_full.min(), y_norm_full.max(), 8)
    # X_train = np.zeros((0, 3))
    # U_train = np.zeros((0, 3))
    # for time in np.unique(X_full[:, 2]):
    #     for x_ in x_int:
    #         for y_ in y_int:
    #             masque_time = X_full[:, 2] == time
    #             distances = np.linalg.norm(
    #                 X_full[masque_time][:, :2] - np.array([x_, y_], dtype=np.float32),
    #                 axis=1,
    #             )
    #             index_min = np.argmin(distances)
    #             point_proche = X_full[masque_time][index_min]
    #             sol_proche = U_full[masque_time][index_min]
    #             X_train = np.concatenate((X_train, point_proche.reshape(-1, 3)))
    #             U_train = np.concatenate((U_train, sol_proche.reshape(-1, 3)))
    mat_data = scipy.io.loadmat("cylinder_Re3900_36points_100snaps.mat")
    data = mat_data["stack"]

    x, y, t = data[:, 0], data[:, 1], data[:, 2]
    x, y = x - x.min(), y - y.min()
    u, v, p = data[:, 3], data[:, 4], data[:, 5]

    x_norm = (x - x.mean()) / x.std()
    y_norm = (y - y.mean()) / y.std()
    t_norm = (t - t.mean()) / t.std()
    p_norm = (p - p.mean()) / p.std()
    u_norm = (u - u.mean()) / u.std()
    v_norm = (v - v.mean()) / v.std()


    X_train = np.array([x_norm, y_norm, t_norm], dtype=np.float32).T

    U_train = np.array([u_norm, v_norm, p_norm], dtype=np.float32).T
        

    mat_data_full = scipy.io.loadmat("cylinder_data.mat")
    data_full = mat_data_full["stack"]

    x_full, y_full, t_full = data_full[:, 0], data_full[:, 1], data_full[:, 2]
    x_full, y_full = x_full - x_full.min(), y_full - y_full.min()
    u_full, v_full, p_full = data_full[:, 3], data_full[:, 4], data_full[:, 5]

    x_norm_full = (x_full - x_full.mean()) / x_full.std()
    y_norm_full = (y_full - y_full.mean()) / y_full.std()
    t_norm_full = (t_full - t_full.mean()) / t_full.std()
    p_norm_full = (p_full - p_full.mean()) / p_full.std()
    u_norm_full = (u_full - u_full.mean()) / u_full.std()
    v_norm_full = (v_full - v_full.mean()) / v_full.std()


    X_full = np.array([x_norm_full, y_norm_full, t_norm_full], dtype=np.float32).T

    U_full = np.array([u_norm_full, v_norm_full, p_norm_full], dtype=np.float32).T

    return X_train, U_train, X_full, U_full


def init_model(f, hyper_param, device, folder_result):
    model = PINNs(hyper_param).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyper_param["lr_init"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=hyper_param["gamma_scheduler"]
    )
    loss = nn.MSELoss()
    # On regarde si notre modèle n'existe pas déjà
    if Path(folder_result + "/model_weights.pth").exists():
        # Charger l'état du modèle et de l'optimiseur
        checkpoint = torch.load(folder_result + "/model_weights.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print("\nModèle chargé\n", file=f)
        print("\nModèle chargé\n")
        csv_train = read_csv(folder_result + "/train_loss.csv")
        csv_test = read_csv(folder_result + "/test_loss.csv")
        train_loss = {
            "total": list(csv_train["total"]),
            "data": list(csv_train["data"]),
            "pde": list(csv_train["pde"]),
        }
        test_loss = {
            "total": list(csv_test["total"]),
            "data": list(csv_test["data"]),
            "pde": list(csv_test["pde"]),
        }
        print("\nLoss chargée\n", file=f)
        print("\nLoss chargée\n")

    else:
        print("Nouveau modèle\n", file=f)
        print("Nouveau modèle\n")
        train_loss = {"total": [], "data": [], "pde": []}
        test_loss = {"total": [], "data": [], "pde": []}
    return model, optimizer, scheduler, loss, train_loss, test_loss


if __name__ == "__main__":
    write_csv([[1, 2, 3], [4, 5, 6]], "ready_cluster/piche/test.csv")
