using DlibDotNet;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System.Drawing;
using System.Linq;

namespace FaceSwapLibrary
{
    public class FaceSwap
    {
       
        /// <summary>
        /// Process the original selfie and produce the face-swapped image.
        /// </summary>
        /// <param name="image">The original selfie image.</param>
        /// <param name="newImage">The new face to insert into the selfie.</param>
        /// <returns>A new image with faces swapped.</returns>
        public static Bitmap ProcessImage(Bitmap image, Bitmap newImage)
        {
            // set up Dlib facedetectors and shapedetectors
            using (FrontalFaceDetector fd = Dlib.GetFrontalFaceDetector())
            using (var sp = ShapePredictor.Deserialize("shape_predictor_68_face_landmarks.dat"))
            {
                // convert image to dlib format
                var img = image.ToArray2D();

                // find bradley's faces in image
                var faces = fd.Operator(img);
                var bradley = faces[0];

                // get bradley's landmark points
                var bradleyShape = sp.Detect(img, bradley);
                var bradleyPoints = (from i in Enumerable.Range(0, (int)bradleyShape.Parts)
                                     let p = bradleyShape.GetPart((uint)i)
                                     select new OpenCvSharp.Point(p.X, p.Y)).ToArray();


                //-------------------------------
                // get convex hull of bradley's points
                var hull = Cv2.ConvexHullIndices(bradleyPoints);
                var bradleyHull = from i in hull
                                  select bradleyPoints[i];


                //---------------------------------
                // find landmark points in face to swap
                var imgMark = newImage.ToArray2D();
                var faces2 = fd.Operator(imgMark);
                var mark = faces2[0];
                var markShape = sp.Detect(imgMark, mark);
                var markPoints = (from i in Enumerable.Range(0, (int)markShape.Parts)
                                  let p = markShape.GetPart((uint)i)
                                  select new OpenCvSharp.Point(p.X, p.Y)).ToArray();

                // get convex hull of mark's points
                var hull2 = Cv2.ConvexHullIndices(bradleyPoints);
                var markHull = from i in hull2
                               select markPoints[i];


                // calculate Delaunay triangles
                var triangles = Utility.GetDelaunayTriangles(bradleyHull);

                // get transformations to warp the new face onto Bradley's face
                var warps = Utility.GetWarps(markHull, bradleyHull, triangles);

                // apply the warps to the new face to prep it for insertion into the main image
                var warpedImg = Utility.ApplyWarps(newImage, image.Width, image.Height, warps);



                //--------------
                // prepare a mask for the warped image
                var mask = new Mat(image.Height, image.Width, MatType.CV_8UC3);
                mask.SetTo(0);
                Cv2.FillConvexPoly(mask, bradleyHull, new Scalar(255, 255, 255), LineTypes.Link8);

                // find the center of the warped face
                var r = Cv2.BoundingRect(bradleyHull);
                var center = new OpenCvSharp.Point(r.Left + r.Width / 2, r.Top + r.Height / 2);

                // blend the warped face into the main image
                var selfie = BitmapConverter.ToMat(image);
                var blend = new Mat(selfie.Size(), selfie.Type());
                Cv2.SeamlessClone(warpedImg, selfie, mask, center, blend, SeamlessCloneMethods.NormalClone);

                // return the modified main image
                return BitmapConverter.ToBitmap(blend);

            }
        }


    }
}
