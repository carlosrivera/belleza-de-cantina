import os
import cv2
import numpy as np
import math
import sys
import dlib
from collections import OrderedDict

def process_landmarks(batch, h, w):
    _, landmarks = process_batch_files(batch, True, False)

    # Eye corners
    eyecornerDst = [(np.int(0.3 * w ), np.int(h / 3)), (np.int(0.7 * w ), np.int(h / 3))]

    pointsNorm = OrderedDict()

    # Add boundary points for delaunay triangulation
    boundaryPts = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2)])

    n = len(list(landmarks.values())[0])

    # Initialize location of average points to 0s
    pointsAvg = np.array([(0,0)] * (n + len(boundaryPts)), np.float32())

    numImages = len(landmarks)

    output = np.zeros((h,w,3), np.float32())
    # Warp images and trasnform landmarks to output coordinate system,
    # and find average of transformed landmarks.

    #for i in range(0, numImages):
    for i, file in enumerate(landmarks):
        print('   - processing avgs({})'.format(i))
        points1 = landmarks[file]

        # Corners of the eye in input image
        eyecornerSrc  = [ landmarks[file][36], landmarks[file][45] ]

        print('   - processing similarityTransform({})'.format(i))
        # Compute similarity transform
        tform = similarityTransform(eyecornerSrc, eyecornerDst)
        # Apply similarity transform on points
        points2 = np.reshape(np.array(points1), (68,1,2))

        points = cv2.transform(points2, tform)

        points = np.float32(np.reshape(points, (68, 2)))

        # Append boundary points. Will be used in Delaunay Triangulation
        points = np.append(points, boundaryPts, axis=0)

        pointsAvg = pointsAvg + points / numImages

        pointsNorm[file] = points

    print(' - calculateDelaunayTriangles({})'.format('batch'))
    # Delaunay triangulation
    rect = (0, 0, w, h)
    dt = calculateDelaunayTriangles(rect, np.array(pointsAvg))
    print(' - dt({})'.format('batch'))

    return landmarks, pointsNorm, pointsAvg, dt


def process_average_batch(batch, landmarks, pointsNorm, pointsAvg, dt, page, h, w):
    images, _ = process_batch_files(batch, False, True)

    # Eye corners
    eyecornerDst = [ (np.int(0.3 * w ), np.int(h / 3)), (np.int(0.7 * w ), np.int(h / 3)) ]

    imagesNorm = OrderedDict()

    # Add boundary points for delaunay triangulation
    boundaryPts = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ])

    # Initialize location of average points to 0s

    n = len(list(landmarks.values())[0])

    numImages = len(images)

    output = np.zeros((h,w,3), np.float32())
    # Warp images and trasnform landmarks to output coordinate system,
    # and find average of transformed landmarks.

    for i, file in enumerate(images):
        # Corners of the eye in input image
        eyecornerSrc  = [landmarks[file][36], landmarks[file][45]]

        print('   - processing similarityTransform({})'.format(page + i))
        # Compute similarity transform
        tform = similarityTransform(eyecornerSrc, eyecornerDst)

        # Apply similarity transformation
        print('   - processing warpAffine({})'.format(page + i))
        img = cv2.warpAffine(images[file], tform, (w,h))
        imagesNorm[file] = img

    # Warp input images to average image landmarks
    for i, file in enumerate(images):
        img = np.zeros((h,w,3), np.float32())
        # Transform triangles one by one
        for j in range(0, len(dt)):
            tin = []
            tout = []

            for k in range(0, 3):
                pIn = pointsNorm[file][dt[j][k]]
                pIn = constrainPoint(pIn, w, h)

                pOut = pointsAvg[dt[j][k]]
                pOut = constrainPoint(pOut, w, h)

                tin.append(pIn)
                tout.append(pOut)

            warpTriangle(imagesNorm[file], img, tin, tout)

        print(' - processing output({})'.format(page + i))
        output = output + img

    return output / numImages


def process_morph_batch(batch, landmarks, pointsNorm, pointsAvg, dt, page, h, w, target):
    images, _ = process_batch_files(batch, False, True)

    # Eye corners
    eyecornerDst = [ (np.int(0.3 * w ), np.int(h / 3)), (np.int(0.7 * w ), np.int(h / 3)) ]

    imagesNorm = OrderedDict()
    images_norm_idx = []


    # Add boundary points for delaunay triangulation
    boundaryPts = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ])

    # Initialize location of average points to 0s

    n = len(list(landmarks.values())[0])

    numImages = len(images)

    output = np.zeros((h,w,3), np.float32())
    # Warp images and trasnform landmarks to output coordinate system,
    # and find average of transformed landmarks.

    for i, file in enumerate(images):
        # Corners of the eye in input image
        eyecornerSrc  = [landmarks[file][36], landmarks[file][45]]

        print('   - processing similarityTransform({})'.format(page + i))
        # Compute similarity transform
        tform = similarityTransform(eyecornerSrc, eyecornerDst)

        # Apply similarity transformation
        print('   - processing warpAffine({})'.format(page + i))
        img = cv2.warpAffine(images[file], tform, (w,h))
        imagesNorm[file] = img
        images_norm_idx.insert(i, file)

    for i, file in enumerate(images_norm_idx):
        if i == len(imagesNorm) -1:
            break


        print('From {} To {}'.format(file, images_norm_idx[(i+1)%len(imagesNorm)]))
        next_file = imagesNorm[images_norm_idx[(i+1)%len(imagesNorm)]]
        print('index: {} {} {}'.format(images_norm_idx[(i+1)%len(imagesNorm)], i+1, (i+1)%len(imagesNorm)))
        for k in range(0, 100, 10):
            y = k / 100.0

            img = (1 - y) * np.asarray(imagesNorm[file]) + y * np.asarray(next_file)
            cv2.imwrite('{}/image_f_{}_{}.jpg'.format(target, page + i, k), img * 255)


    return output / numImages


def process_batch_files(batch, process_landmarks = False, process_images = False):
    p = "/root/shape_predictor_68_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    landmarks = OrderedDict()
    images = OrderedDict()

    for filePath in batch:
        print('  - reading file: {}, processing: landmarks({}), images({})'.format(filePath, process_landmarks, process_images))
        # Read image found.
        img = cv2.imread(filePath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = detector(gray, 0)

        maxArea = 0
        x = 0
        y = 0
        w = 0
        h = 0


        for (i, rect) in enumerate(rects):
            if rect.area() > maxArea:
                x = rect.left()
                y = rect.top()
                w = rect.width()
                h = rect.height()
                maxArea = rect.area()

        if maxArea:

            shape = predictor(gray, dlib.rectangle(x, y, x+w, y+h))

            if shape:
                if process_landmarks:
                    landmarks[filePath] = np.array([(shape.part(i).x, shape.part(i).y) for i in range(0, 68)])


                if process_images:
                    images[filePath] = np.float32(img) / 255.0

        # loop over the face detections
        #for (i, rect) in enumerate(rects):


    return images, landmarks

# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.

def similarityTransform(inPoints, outPoints):
    s60 = math.sin(60*math.pi/180)
    c60 = math.cos(60*math.pi/180)

    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()

    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]

    inPts.append([np.int(xin), np.int(yin)])

    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]

    outPts.append([np.int(xout), np.int(yout)])

    tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]), False)

    return tform[0]


# Check if a point is inside a rectangle
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

# Calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # Create subdiv
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]))

    #subdiv.insert([(p[0], p[1]) for p in points])

    # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
    triangleList = subdiv.getTriangleList()

    # Find the indices of triangles in the points array

    delaunayTri = []

    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

    return delaunayTri


def constrainPoint(p, w, h):
    p =  ( min( max( p[0], 0 ) , w - 1 ) , min( max( p[1], 0 ) , h - 1 ) )
    return p

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )

    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect

def get_files(path, batch_size):
    files = os.listdir(path)
    files.sort()

    files = [os.path.join(path, e) for e in files if e.endswith('.jpg')]
    batched_files = [files[e:e + batch_size] for e in range(0, len(files), batch_size)]

    return files, batched_files

def face_average(path, target, w = 600, h = 600, batch_size = 2):
    files, batched_files = get_files(path, batch_size)

    output = np.zeros((w , h, 3), np.float32())

    landmarks, normals, avgs, dt = process_landmarks(files, w, h)

    for current_batch, batch in enumerate(batched_files):
        print('Processing batch: {}'.format(current_batch))

        output = output + process_average_batch(batch, landmarks, normals, avgs, dt, current_batch * batch_size, w, h)

        cv2.imwrite('batch_{}.jpg'.format(current_batch), (output / (batch_size * (max(current_batch, 1)))) * 255)

    # Display result
    cv2.imwrite(target, (output / len(batched_files)) * 255)

def face_morph(path, target, w = 600, h = 600, batch_size = 2):
    files, batched_files = get_files(path, batch_size)

    output = np.zeros((w , h, 3), np.float32())

    landmarks, normals, avgs, dt = process_landmarks(files, w, h)

    for current_batch, batch in enumerate(batched_files):
        print('Processing batch: {}'.format(current_batch))

        if (current_batch > 0):
            print('Inserted previous file: {}'.format(batched_files[current_batch - 1][-1]))
            batch.insert(0, batched_files[current_batch - 1][-1])

        print(batch)
        process_morph_batch(batch, landmarks, normals, avgs, dt, current_batch * batch_size, w, h, target)




if __name__ == '__main__':

    #face_average('me/', 'final.jpg')
    face_morph('me/', 'me_int')
