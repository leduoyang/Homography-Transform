import numpy as np
import cv2
import pdb
from matplotlib import pyplot as plt
import pdb

# u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
    A = np.zeros((2*N, 8))
    b = np.zeros((2*N, 1))
    A[0:A.shape[0]:2,2] = 1
    A[1:A.shape[0]:2,5] = 1
    for n in range(N):
        r = n * 2
        A[r,0] = u[n][0]
        A[r,1] = u[n][1]
        A[r,6] = -1*(u[n][0]*v[n][0])
        A[r,7] = -1*(u[n][1]*v[n][0])
        r = r+1
        A[r,3] = u[n][0]
        A[r,4] = u[n][1]
        A[r,6] = -1*(u[n][0]*v[n][1])
        A[r,7] = -1*(u[n][1]*v[n][1])
    for n in range(N):
        r = n * 2
        b[r,0] = v[n][0]
        r = r + 1
        b[r,0] = v[n][1]
    
    X = np.linalg.solve(A, b)    
    H = np.zeros((3, 3))
    H[2,2] = 1
    for x in range(len(X)) :
        r = x // 3
        c = x % 3
        H[r,c] = X[x]
    return H


# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    h, w, ch = img.shape
    # TODO: some magic


def main():
    ####################################### Part 1 #######################################
    canvas = cv2.imread('./input/times_square.jpg')
    h,w,ch = canvas.shape
    size = tuple([w,h])
    
    img1 = cv2.imread('./input/wu.jpg')
    img2 = cv2.imread('./input/ding.jpg')
    img3 = cv2.imread('./input/yao.jpg')
    img4 = cv2.imread('./input/kp.jpg')
    img5 = cv2.imread('./input/lee.jpg')
    # order : 左上,左下,右上,右下
    ref_corners1 = np.array([[818, 352], [884, 352], [818, 407], [885, 408]]) 
    ref_corners2 = np.array([[311, 14], [402, 150], [157, 152], [278, 315]]) 
    ref_corners3 = np.array([[364, 674], [430, 725], [279, 864], [369, 885]])
    ref_corners4 = np.array([[808, 495], [892, 495], [802, 609], [896, 609]])
    ref_corners5 = np.array([[1024, 608], [1118, 593], [1032, 664], [1134, 651]])
    V = np.concatenate([ref_corners1,ref_corners2,ref_corners3,ref_corners4,ref_corners5],axis=0)
    
    org_corners1 = np.array([[0,0],[img1.shape[0]-1,0],[0,img1.shape[1]-1],[img1.shape[0]-1,img1.shape[1]-1]])
    org_corners2 = np.array([[0,0],[img2.shape[0]-1,0],[0,img2.shape[1]-1],[img2.shape[0]-1,img2.shape[1]-1]])
    org_corners3 = np.array([[0,0],[img3.shape[0]-1,0],[0,img3.shape[1]-1],[img3.shape[0]-1,img3.shape[1]-1]])
    org_corners4 = np.array([[0,0],[img4.shape[0]-1,0],[0,img4.shape[1]-1],[img4.shape[0]-1,img4.shape[1]-1]])
    org_corners5 = np.array([[0,0],[img5.shape[0]-1,0],[0,img5.shape[1]-1],[img5.shape[0]-1,img5.shape[1]-1]])
    U = np.concatenate([org_corners1,org_corners2,org_corners3,org_corners4,org_corners5],axis=0) 
    
    img_set = {'img1':img1,'img2':img2,'img3':img3,'img4':img4,'img5':img5}
    img_key = ['img1','img2','img3','img4','img5']
    for n in range(5):
        r = n * 4
        u = U[r:r+4,:]
        v = V[r:r+4,:]
        H = solve_homography(u, v)
        #H, status = cv2.findHomography(u, v)
        img_in_canvas = cv2.warpPerspective(img_set[img_key[n]], H , size)
        indices = np.nonzero(img_in_canvas)
        canvas[indices[0],indices[1],:] = 0
        canvas = canvas + img_in_canvas
          
    plt.figure(0)
    plt.imshow(canvas)
    plt.show()     
    cv2.imwrite('part1.png', canvas)

    ####################################### Part 2 #######################################
    img = cv2.imread('./input/screen.jpg')
    # order : 左上,左下,右上,右下
    tl,bl,tr,br = [1038,365],[980,553],[1106,393],[1041,594]
    org_corners  = np.array([tl,bl,tr,br],np.float32)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    size = int(max(widthA,widthB,heightA,heightB))
    ref_corners = np.array([[0, 0],[size - 1, 0],[0, size - 1],[size - 1, size - 1]],np.float32)
    H = solve_homography(org_corners, ref_corners)
    warped = cv2.warpPerspective(img, H, (size, size),flags=cv2.INTER_LINEAR)
    
    plt.figure(0)
    plt.imshow(warped)
    plt.show()    
    output2 = warped
    cv2.imwrite('part2.png', output2)

    # Part 3
    img_front = cv2.imread('./input/crosswalk_front.jpg')
    h,w,c = img_front.shape
    # order : top-left,bottom-left,top-right,bottom-right
    
    #region containing all the bars
    tl,bl,tr,br = [136,162],[64,238],[588,158],[661,260]
    org_corners  = np.array([tl,bl,tr,br],np.float32)
    ref_corners = np.array([[80, 80],[96, 254],[422,80],[409,255]],np.float32) 
    M = cv2.getPerspectiveTransform(org_corners,ref_corners)
    warped = cv2.warpPerspective(img_front, M, (498, 355))
    
    #first
    tl,bl,tr,br = [136,162],[64,270],[174,162],[109,265]
    org_corners1  = np.array([tl,bl,tr,br],np.float32)
    ref_corners1 = np.array([[80, 80],[96, 254],[107,80],[119,252]],np.float32)     
    #middle
    tl,bl,tr,br = [345,157],[340,260],[381,157],[386,260]
    org_corners2  = np.array([tl,bl,tr,br],np.float32)
    ref_corners2 = np.array([[237, 80],[237, 254],[265,80],[262,254]],np.float32)    
    #last
    tl,bl,tr,br = [552,157],[616,260],[588,158],[661,260]
    org_corners3  = np.array([tl,bl,tr,br],np.float32)
    ref_corners3 = np.array([[395, 80],[385, 254],[422,80],[409,255]],np.float32)    
    M1 = cv2.getPerspectiveTransform(org_corners1,ref_corners1)
    M2 = cv2.getPerspectiveTransform(org_corners2,ref_corners2)
    M3 = cv2.getPerspectiveTransform(org_corners3,ref_corners3)
    M = M2
    warped = cv2.warpPerspective(img_front, M, (498, 355))
    
    
    plt.figure(0)
    plt.imshow(warped)
    plt.show()    
    output3 = warped
    cv2.imwrite('part3.png', output3)


if __name__ == '__main__':
    main()



































