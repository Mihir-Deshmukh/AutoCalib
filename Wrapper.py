import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize as opt
import copy
 
def get_image_corners(images, pattern_size):
    image_corners = []
    for i, image in enumerate(images):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            image_corners.append(corners.squeeze())
            image_with_corners = cv2.drawChessboardCorners(image.copy(), pattern_size, corners, ret)
            cv2.imwrite(f'./mpdeshmukh_hw1/ReprojectionResults/Corners/ChessboardCorners_{i}.png', image_with_corners)
    
            # plt.imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))
            # plt.title('Chessboard Corners')
            # plt.axis('off')
            # plt.show()
        else:
            print('Chessboard corners not found in the image')
    return np.array(image_corners)

def calculate_homography(image_corners, world_corners):
    H = []
    for i in range(len(image_corners)):
        # All corners of one image instance
        homography, _ = cv2.findHomography(world_corners, image_corners[i])
        H.append(homography)
    return H

def calculate_V(H, i, j):
    V = np.array([H[0][i] * H[0][j], 
                  H[0][i] * H[1][j] + H[1][i] * H[0][j], 
                  H[1][i] * H[1][j],
                  H[2][i] * H[0][j] + H[0][i] * H[2][j],
                  H[2][i] * H[1][j] + H[1][i] * H[2][j],
                  H[2][i] * H[2][j]
                  ])
    return V.T


def calculate_b(H):
    V = []
    for homography in H:
        
        v11 = calculate_V(homography, 0, 0)
        v12 = calculate_V(homography, 0, 1)
        v22 = calculate_V(homography, 1, 1)
        
        v_image = np.vstack((v12.T, (v11 - v22).T))
        
        # print(f"V_image shape: {v_image.shape}")
        V.append(v_image)
        
    V = np.vstack(V)
    # V = V.reshape(-1, 6)
    print(f"V shape: {V.shape}")
    
    _, _, Vt = np.linalg.svd(V)
    b = Vt.T[:, -1]
    B = np.array([[b[0], b[1], b[3]], [b[1], b[2], b[4]], [b[3], b[4], b[5]]])
    print(f"B shape: {b}")
    
    return B
    

def calculate_intrinsics(H):
    # B = K*K-1
    # print(len(H))
    B = calculate_b(H)
    print(B)
    
    v0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - B[0, 1] ** 2)
    lambDa = B[2, 2] - (B[0, 2] ** 2 + v0 * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]
    alpha = np.sqrt(lambDa / B[0, 0])
    beta = np.sqrt(lambDa * B[0, 0] / (B[0, 0] * B[1, 1] - B[0, 1] ** 2))
    gamma = -B[0, 1] * alpha ** 2 * beta / lambDa
    u0 = (gamma * v0 / beta) - (B[0, 2] * alpha ** 2 / lambDa)
    K = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
    
    # print(f"lambda: {lambDa}")
    # print(f"alpha: {alpha}")
    # print(f"beta: {beta}")
    # print(f"gamma: {gamma}")
    # print(f"u0: {u0}")
    # print(f"v0: {v0}")
    
    return K

def calculate_extrinsics(K, H):
    
    Rts = []
    for homography in H:
        h1 = homography[:, 0]
        h2 = homography[:, 1]
        h3 = homography[:, 2]
        
        lambDa = 1 / np.linalg.norm(np.dot(np.linalg.inv(K), h1))
        # print(f"Lambda: {lambDa}")
        # lambDa = 1 / np.linalg.norm(np.dot(np.linalg.inv(K), h2))
        # print(f"Lambda: {lambDa}")
        r1 = lambDa * np.dot(np.linalg.inv(K), h1)
        r2 = lambDa * np.dot(np.linalg.inv(K), h2)
        r3 = np.cross(r1, r2) #Not used as is multiplied by 0 later on
        t = lambDa * np.dot(np.linalg.inv(K), h3)
        
        Rt = np.column_stack((r1, r2, r3, t))
        
        Rts.append(Rt)

    return Rts

def calculate_reprojection_error(x0, Rts, world_corners, image_corners, ret = False):
    fx = x0[0]
    fy = x0[2]
    u0 = x0[3]
    v0 = x0[4]
    gamma = x0[1]
    k1 = x0[5]
    k2 = x0[6]
    error_all = 0
    reproj_corners_all = []
    # print(f"image_corners: {image_corners.shape}")
    # print(f"world_corners: {world_corners.shape}")
    
    K = np.array([[fx, gamma, u0], [0, fy, v0], [0, 0, 1]])
    # print(f"K: {K}")
    
    for i, img_corners in enumerate(image_corners):
        
        # img_corners = image_corners[i]
        Rt = Rts[i]
        H = np.dot(K, Rt)
        error_img = 0
        reproj_corners = []
        
        for j, _ in enumerate(world_corners):
            
            # print(img_corners.shape)
            
            imageCorners = np.append(img_corners[j,:], 1)
            worldCorners = np.append(world_corners[j], 1)
            # print(f"World Corner: {imageCorners}")
            
            # print(f"Image Corner: {imageCorners}")
            
            camera_frame_coords = Rt @ worldCorners
            # print(f"\nCamera Frame Coords: {camera_frame_coords}")
            
            x = camera_frame_coords[0] / camera_frame_coords[2]
            y = camera_frame_coords[1] / camera_frame_coords[2]
            
            image_pixel = H @ worldCorners
            # print(f"Image Pixel: {image_pixel}")
            
            u = image_pixel[0] / image_pixel[2]
            v = image_pixel[1] / image_pixel[2]
            
            # print(f"u: {u}")
            # print(f"v: {v}")
            
            u_hat = u + (u - u0) * (k1 * (x ** 2 + y ** 2) + k2 * (x ** 2 + y ** 2) ** 2)
            v_hat = v + (v - v0) * (k1 * (x ** 2 + y ** 2) + k2 * (x ** 2 + y ** 2) ** 2)
            
            corners_hat = np.array([u_hat, v_hat, 1])
            # print(f"Reprojected Corners: {corners_hat}")
            # print(f"Image Corners: {img_corners[j,:]}")
            reproj_corners.append(corners_hat[:2])
            
            error = np.linalg.norm(corners_hat - imageCorners, 2)
            error_img += error
            
        
        # print(f"Error for Image {i}: {error_img/len(world_corners)}")
        # error_all.append(error_img/len(world_corners))
        error_all += error_img/len(world_corners)
        reproj_corners_all.append(reproj_corners)
    
    # print(f"Error All: {error_all}")
    if ret:
        return error_all, np.array(reproj_corners_all)
    
    return np.array([error_all, 0, 0, 0, 0, 0, 0])


def plot_reprojection(image, corners, image_number):
    img = copy.deepcopy(image)
    corners = corners.reshape(-1, 2)
    # print(corners.shape)
    for i in range(corners.shape[0]):
        # print(corners[i][0], corners[i][1])
        cv2.circle(img, (int(corners[i][0]),int(corners[i][1])), 7, (0,0,255), -1)
    cv2.imwrite(f"./mpdeshmukh_hw1/ReprojectionResults/{image_number}.png", img)
            
def main():

    folder_path = './Calibration_Imgs'
    images = []
    image_names = []
    
    if not os.path.exists('./mpdeshmukh_hw1/ReprojectionResults/Corners'):
        os.makedirs('./mpdeshmukh_hw1/ReprojectionResults/Corners')

    # Read images from the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image = cv2.imread(os.path.join(folder_path, filename))
            images.append(image)
            image_names.append(filename)
            # print(f"Image: {filename}")

    # Size of the chessboard (Grid size) and dimension in mm
    pattern_size = (9, 6)
    square_size = 21.5
    
    # Get Checkerboard Corners
    image_corners = get_image_corners(images, pattern_size)
    print(f"Image Corners: {image_corners.shape}")
    
    # World Corners    
    x, y = np.meshgrid(range(pattern_size[0]), range(pattern_size[1]))
    world_corners = np.hstack((x.reshape(-1, 1) * square_size, y.reshape(-1, 1) * square_size, np.zeros((pattern_size[0] * pattern_size[1], 1)))).astype(np.float32)
    print(f"World Corners: {world_corners.shape}")
    # print(world_corners)
    
    
    H = calculate_homography(image_corners, world_corners)
    print(f"Homography shape: {np.array(H).shape}")
    # print(f"Homography: {H}")


    # Calculate Intrinsic Parameters
    K = calculate_intrinsics(H)
    print(f"\nIntrinsic Parameters K:\n {K}")
    
    Rts = calculate_extrinsics(K, H)
    print(f"\nExtrinsic Parameters Rt for Image 0:\n {Rts[0]}")
    print(f"Extrinsic Parameters Shape: \n {np.array(Rts).shape}")
    
    k = np.array([0, 0])
    parameters = np.array([K[0,0], K[0,1], K[1,1], K[0,2], K[1,2], k[0], k[1]])
    
    print(f"\nInitial Parameters: {parameters}")
    print(f"Optimizing Parameters...")
    x = opt.least_squares(fun=calculate_reprojection_error, x0=parameters, method='lm', args=(Rts, world_corners, image_corners), max_nfev=10000)
    print(f"Optimized Parameters: {x.x}")
    fx, gamma, fy, u0, v0, k1, k2 = x.x
    K_new = np.array([[fx, gamma, u0], [0, fy, v0], [0, 0, 1]])
    K_distortion_new = np.array([k1, k2, 0, 0, 0], dtype = float)
    
    print(f"\nNew Intrinsic Parameters K:\n {K_new}")
    print(f"Optimized distortion parameters: {k1}, {k2}")
    
    errors = []
    # print(image_corners[0].reshape(1, -1, 2).shape)
    for i, image in enumerate(images):
        error, reproj_corners = calculate_reprojection_error(x.x, [Rts[i]], world_corners, image_corners[i].reshape(1, -1, 2), ret=True)
        print(f"Reprojection Error for Image {i}: {error}")
        img = cv2.undistort(image, K_new, K_distortion_new)
        # print(reproj_corners)
        plot_reprojection(img, reproj_corners, i)
        errors.append(error)

    print("Mean Reprojection error", np.mean(np.array(errors)))
    
    

if __name__ == '__main__':
    main()