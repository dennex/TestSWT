using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace testSWT
{
	public class Point2d{
		public int x;
		public int y;
		public float SWT;

        public override string ToString()
        {
            return "x:"+x+" y:"+y+" SWT:"+SWT;
        }
	}

	public struct Point2dFloat
	{
		public float x;
		public float y;
	}

	public struct Point3dFloat
	{
		public float x;
		public float y;
		public float z;
	}

	public class Ray
	{
		public Point2d p;
		public Point2d q;
		public List<Point2d> points;
		public bool valid;
	}

	public class Rays
	{
		public float median;
		public float variance;
		public float stdev;
		public float average;
		public List<double> rayLengths;
		public List<Ray> rays;

		public Rays(List<Ray> rays)
		{
			this.rays = new List<Ray>( rays );
			rayLengths = new List<double>();
			
			foreach (Ray ray in rays)
			{
				if( ray.p.x == ray.q.x && ray.p.y == ray.q.y )
					rayLengths.Add( 0 );
				else
					rayLengths.Add( Math.Sqrt( Math.Pow( ray.p.x - ray.q.x,2 )  + Math.Pow( ray.p.y - ray.q.y,2 ) ));
			}

			this.GetStats();
		}

		public void SetValid(bool valid)
		{
			for( int i = 0; i < this.rays.Count();i++ )
			{
				this.rays[i].valid = valid;
			}

		}

		public void GetStats()
		{			

			average = (float)rayLengths.Average();
			variance = ( float )GetVariance( rayLengths );
			stdev = (float)Math.Sqrt( variance );
			List<double> rayLengthsCopy = new List<double>( rayLengths );
			rayLengthsCopy.Sort();
			median = (float)rayLengthsCopy[ rayLengthsCopy.Count() / 2 ];


		}

		public static double GetVariance(List<double> rayLengths)
		{
			if( rayLengths.Count() <= 1 )
				return 0;
			else
			{
				double average = rayLengths.Average();
				double sum = 0;
				foreach( double r in rayLengths )
				{
					sum += Math.Pow( r - average, 2 );
				}

				return sum / ( rayLengths.Count() - 1 );
			}
		}
	}

	public struct Chain
	{
		public int p;
		public int q;
		public float dist;
		public bool merged;
		public Point2dFloat direction;
		public List<int> components;
	}

	public class Graph:Dictionary<int,GraphNode>
	{
		public Graph()
		{

		}

		public void Add(int key,int value)
		{
			if (this.ContainsKey(key))
			{
                this[key].adjacency.Add(value);
				
			}
			else
			{
                GraphNode gn = new GraphNode(key);
                gn.adjacency.Add(value);
				base.Add(key,gn);
			}
		}

        public void Add(int key)
        {
            if (this.ContainsKey(key))
            {
                // do nothing
            }
            else
            {
                GraphNode gn = new GraphNode(key);
                base.Add(key, gn);
            }
        }

        public void ResetVisited()
        {
            foreach (KeyValuePair<int,GraphNode> graphNode in this)
            {
                graphNode.Value.visited = false;
            }
        }

        public GraphNode NextUnvisited()
        {
            foreach (KeyValuePair<int,GraphNode> graphNode in this)
            {
                if (graphNode.Value.visited == false)
                    return graphNode.Value;

            }

            return null;
        }

	}

    public class GraphNode
    {
        public int vertexValue;
        public List<int> adjacency;
        public bool visited = false;

        public GraphNode()
        {
        }

        public GraphNode(int key)
        {
            vertexValue = key;
            adjacency = new List<int>();
            visited = false;
        }
    }


	public class Point2dComparer:IComparer<Point2d>
	{
		public int Compare(Point2d lhs, Point2d rhs)
		{
			if (lhs.SWT == rhs.SWT)
			{
				return 0;
			}
			else if (lhs.SWT > rhs.SWT)
			{
				return 1;
			}
			else
			{
				return -1;
			}
		}
	}

	public class SWT
	{
		public List<Tuple<CvPoint,CvPoint>> FindBoundingBoxes(List<List<Point2d>> components, List<Chain> chains, List<Tuple<Point2d,Point2d>> compBB, IplImage output)
		{
			
			List<Tuple<CvPoint, CvPoint>> bb = new List<Tuple<CvPoint,CvPoint>>();

			bb.Capacity = chains.Count();
			foreach (Chain chain in chains)
			{
				int minx = output.Width;
				int miny = output.Height;
				int maxx = 0;
				int maxy = 0;

				foreach (int cit in chain.components)
				{
					miny = Math.Min(miny,compBB[cit].Item1.y);
					minx = Math.Min(minx,compBB[cit].Item1.x);
					maxy = Math.Max(maxy,compBB[cit].Item2.y);
					maxx = Math.Max(maxx,compBB[cit].Item2.x);
				}

				CvPoint p0 = new CvPoint(minx, miny);
				CvPoint p1 = new CvPoint(maxx, maxy);
				bb.Add(new Tuple<CvPoint,CvPoint>(p0,p1));

			}


			return bb;
		}

		public List<Tuple<CvPoint, CvPoint>> FindBoundingBoxes( List<List<Point2d>> components, IplImage output )
		{

			List<Tuple<CvPoint, CvPoint>> bb = new List<Tuple<CvPoint, CvPoint>>();

			bb.Capacity = components.Count();
			foreach( List<Point2d> chain in components )
			{
				int minx = output.Width;
				int miny = output.Height;
				int maxx = 0;
				int maxy = 0;

				foreach( Point2d cit in chain )
				{
					miny = Math.Min( miny, cit.y );
					minx = Math.Min( minx, cit.x );
					maxy = Math.Max( maxy, cit.y );
					maxx = Math.Max( maxx, cit.x );
				}

				CvPoint p0 = new CvPoint( minx, miny );
				CvPoint p1 = new CvPoint( maxx, maxy );
				bb.Add( new Tuple<CvPoint, CvPoint>( p0, p1 ) );

			}
			return bb;
		}

		public static void NormalizeImage (IplImage input, IplImage output)
		{
			double maxVal;
			double minVal;
			Cv.MinMaxLoc( input, out minVal, out maxVal );

			Cv.Normalize( input, output, maxVal, minVal );
			
		}

        public static void ConvertColorHue(IplImage origImage, IplImage convertedImage)
        {
            IplImage LABimage = Cv.CreateImage(origImage.GetSize(),BitDepth.U8, 3);
            Cv.CvtColor(origImage, LABimage, ColorConversion.BgrToLab);
            origImage.SetROI(new CvRect(195, 121, 6,7));
            IplImage whiteBlock = Cv.CreateImage(origImage.GetSize(),BitDepth.U8,3);
            Cv.Copy(origImage, whiteBlock);
            IplImage whiteLAB = Cv.CreateImage(whiteBlock.GetSize(), BitDepth.U8, 3);
            Cv.CvtColor(whiteBlock, whiteLAB, ColorConversion.BgrToLab);
            origImage.ResetROI();

            unsafe
            {
                byte* whitePtr = (byte*)whiteLAB.ImageData.ToPointer();
                int whiteWidthStep = whiteLAB.WidthStep;
                int count = whiteLAB.Height * whiteLAB.Width;
                double L = 0, A = 0, B = 0;
                for (int i = 0; i < whiteLAB.Height; i++)
                {
                    for (int j = 0; j < whiteLAB.Width; j++)
                    {
                        L += whitePtr[i*whiteWidthStep + 3*j];
                        A += whitePtr[i*whiteWidthStep + 3*j+1];
                        B += whitePtr[i * whiteWidthStep + 3*j+2];
                    }
                }
                L = L / count;
                A = A / count;
                B = B / count;


                byte* convertedPtr = (byte*)convertedImage.ImageData.ToPointer();
                int convertedWidthStep = convertedImage.WidthStep;

                byte* LABPtr = (byte*)LABimage.ImageData.ToPointer();
                int LABWidthStep = LABimage.WidthStep;

                double Luminance = 0, Alpha = 0, Beta = 0;
                for (int i = 0; i < convertedImage.Height; i++)
                {
                    for (int j = 0; j < convertedImage.Width; j++)
                    {
                        Luminance = LABPtr[i * LABWidthStep + 3*j] - L;
                        Alpha = LABPtr[i * LABWidthStep + 3*j + 1] - A;
                        Beta = LABPtr[i * LABWidthStep + 3*j + 2] - B;
                        convertedPtr[i * convertedWidthStep + j] = (byte)(Math.Sqrt(Luminance*Luminance + Alpha*Alpha + Beta*Beta)/Math.Sqrt(3));
                    }
                }

            }


        }

		public static IplImage textDetection (IplImage input, bool darkOnLight)
		{
            darkOnLight = true;
			// convert to grayscale, could do better with color subtraction if we know what white will look like
            IplImage grayImg = Cv.CreateImage( input.GetSize(), BitDepth.U8, 1 );
			//Cv.CvtColor( input, grayImg, ColorConversion.BgrToGray );
            ConvertColorHue(input, grayImg);

			// create canny --> hard to automatically find parameters...
			double threshLow = 10;//5
			double threshHigh = 30;//50
			IplImage edgeImg = Cv.CreateImage( input.GetSize(), BitDepth.U8, 1 );
			Cv.Canny( grayImg, edgeImg, threshLow, threshHigh,ApertureSize.Size3 );
			Cv.SaveImage( "canny.png", edgeImg );

			// create gradient x, gradient y
			IplImage gaussianImg = Cv.CreateImage( input.GetSize(), BitDepth.F32, 1 );
			Cv.ConvertScale( grayImg, gaussianImg, 1.0 / 255.0, 0 );
			Cv.Smooth( gaussianImg, gaussianImg, SmoothType.Gaussian, 5, 5 );
			IplImage gradientX = Cv.CreateImage( input.GetSize(), BitDepth.F32, 1 );
			IplImage gradientY = Cv.CreateImage( input.GetSize(), BitDepth.F32, 1 );
			Cv.Sobel(gaussianImg, gradientX,1,0,ApertureSize.Scharr);
			Cv.Sobel(gaussianImg, gradientY,0,1,ApertureSize.Scharr);
			Cv.Smooth( gradientX, gradientX,SmoothType.Blur,3,3 );
			Cv.Smooth( gradientY, gradientY,SmoothType.Blur,3, 3 );


			// calculate SWT and return ray vectors
			List<Ray> rays = new List<Ray>();
			IplImage SWTImage = Cv.CreateImage( input.GetSize(), BitDepth.F32, 1 );
			SWTImage.Set( -1 );

			// stroke width transform
			strokeWidthTransform( edgeImg, gradientX, gradientY, darkOnLight, SWTImage, rays );

            // swtmedianfilter
			SWTMedianFilter( SWTImage, rays );

            // not in the original algorithm... if rays are deviating too much from median, remove
			IplImage cleanSWTImage = Cv.CreateImage( input.GetSize(), BitDepth.F32, 1 );
			cleanSWTImage.Set( -1 );
			FilterRays( SWTImage, rays, cleanSWTImage );

            // normalize
			IplImage output2 = Cv.CreateImage( input.GetSize(), BitDepth.F32, 1 );
            NormalizeImage(SWTImage, output2);

            // binarize and close with rectangle to fill gaps from cleaning
			IplImage binSWTImage = Cv.CreateImage( cleanSWTImage.GetSize(), BitDepth.U8, 1 );
			IplImage tempImg = Cv.CreateImage( cleanSWTImage.GetSize(), BitDepth.U8, 1 );
			Cv.Threshold( cleanSWTImage, binSWTImage, 1, 255, ThresholdType.Binary );
			Cv.MorphologyEx( binSWTImage, binSWTImage, tempImg, new IplConvKernel(5, 17, 2, 8, ElementShape.Rect ), MorphologyOperation.Close );

			IplImage saveSWT = Cv.CreateImage( input.GetSize(), BitDepth.U8, 1 );
			Cv.ConvertScale( output2, saveSWT, 255, 0 );
			Cv.SaveImage( "SWT.png", saveSWT );

			

			//// Calculate legally connect components from SWT and gradient image.
			//// return type is a vector of vectors, where each outer vector is a component and
			//// the inner vector contains the (y,x) of each pixel in that component.
			//List<List<Point2d>> components = findLegallyConnectedComponents( SWTImage, rays );

			List<List<Point2d>> components = FindLegallyConnectedComponents( cleanSWTImage, rays );

            IplImage binFloatImg = Cv.CreateImage(binSWTImage.GetSize(), BitDepth.F32, 1);
            Cv.Convert(binSWTImage, binFloatImg);
            List<List<Point2d>> components2 = FindLegallyConnectedComponents(binFloatImg, rays);
			// Filter the components

			List<List<Point2d>> validComponents = new List<List<Point2d>>();
			List<Tuple<Point2d, Point2d>> compBB = new List<Tuple<Point2d,Point2d>>();

			List<Point2dFloat> compCenters = new List<Point2dFloat>();
			List<float> compMedians = new List<float>();
			List<Point2d> compDimensions = new List<Point2d>();

            FilterComponents(cleanSWTImage, components, ref validComponents, ref compCenters, ref compMedians, ref compDimensions, ref compBB);

			IplImage output3 = Cv.CreateImage( input.GetSize(), BitDepth.U8, 3 );

            RenderComponentsWithBoxes(cleanSWTImage, components, compBB, output3);
			Cv.SaveImage( "components.png", output3 );
			////cvReleaseImage ( &output3 );

			//// Make chains of components
			//List<Chain> chains;

			//chains = makeChains( input, validComponents, compCenters, compMedians, compDimensions, compBB );

			//IplImage output4 = Cv.CreateImage( input.GetSize(), BitDepth.U8, 1 );
			
			//renderChains( SWTImage, validComponents, chains, output4 );
			////cvSaveImage ( "text.png", output4);

			//IplImage output5 = Cv.CreateImage( input.GetSize(), BitDepth.U8, 3 );
			//Cv.CvtColor( output4, output5,ColorConversion.GrayToRgb );

            return output3;
		}

        public static void RenderComponentsWithBoxes(IplImage SWTImage, List<List<Point2d>> components, List<Tuple<Point2d, Point2d>> compBB, IplImage output)
        {
            IplImage outTemp = Cv.CreateImage(output.GetSize(), BitDepth.F32, 1);
            RenderComponents(SWTImage, components, outTemp);
            List<Tuple<CvPoint, CvPoint>> bb = new List<Tuple<CvPoint, CvPoint>>(compBB.Count());

            foreach (Tuple<Point2d, Point2d> it in compBB)
            {
                CvPoint p0 = new CvPoint(it.Item1.x, it.Item1.y);
                CvPoint p1 = new CvPoint(it.Item2.x, it.Item2.y);
                Tuple<CvPoint, CvPoint> pair = new Tuple<CvPoint,CvPoint>(p0,p1);
                bb.Add(pair);
            }

            IplImage outImg = Cv.CreateImage(output.GetSize(), BitDepth.U8, 1);
			
            Cv.Convert(outTemp, outImg);
            Cv.CvtColor(outImg, output, ColorConversion.GrayToBgr);

            int count = 0;
            foreach (Tuple<CvPoint, CvPoint> it in bb)
            {
                CvScalar c;
                if (count % 3 == 0) c = new CvScalar(255, 0, 0);
                else if (count % 3 == 1) c = new CvScalar(0, 255, 0);
                else c = new CvScalar(0, 0, 255);
                count++;
                Cv.Rectangle(output, it.Item1, it.Item2, c, 2);
            }
        }

        public static void RenderComponents(IplImage SWTImage, List<List<Point2d>> components, IplImage output)
        {
            Cv.Zero(output);
            unsafe
            {
                float* swtPtr = (float*)SWTImage.ImageData.ToPointer();
                int swtWidthStep = SWTImage.WidthStep / 4;

                float* outPtr = (float*)output.ImageData.ToPointer();
                int outWidthStep = output.WidthStep / 4;

                foreach (List<Point2d> it in components)
                {
                    foreach (Point2d pixel in it)
                    {
                        outPtr[pixel.y * outWidthStep + pixel.x] = swtPtr[pixel.y * swtWidthStep + pixel.x];
                    }
                }

                for (int row = 0; row < output.Height; row++)
                {
                    for (int col = 0; col < output.Width; col++)
                    {
                        if (outPtr[row * outWidthStep + col] == 0)
                        {
                            outPtr[row * outWidthStep + col] = -1;
                        }
                    }
                }
                double maxVal;
				double minVal;

				Cv.MinMaxLoc( output, out minVal, out maxVal );

				float minFloat = (float)minVal;
				float maxFloat = ( float )maxVal;

				float difference = ( maxFloat - minFloat );
                for (int row = 0; row < output.Height; row++)
                {
                    for (int col = 0; col < output.Width; col++)
                    {
                        if (outPtr[row * outWidthStep + col] < 1)
                        {
                            outPtr[row * outWidthStep + col] = 1;
                        }
                        else
                        {
							outPtr[ row * outWidthStep + col ] = ( ( outPtr[ row * outWidthStep + col ] ) - minFloat ) / difference * 255;
                        }
                    }
                }
            }
        }

        public static void FilterComponents(IplImage SWTImage,
                                            List<List<Point2d>> components,
                                            ref List<List<Point2d>> validComponents,
                                            ref List<Point2dFloat> compCenters,
                                            ref List<float> compMedians,
                                            ref List<Point2d> compDimensions,
                                            ref List<Tuple<Point2d, Point2d>> compBB)
        {
            validComponents = new List<List<Point2d>>(components.Count());
            compCenters = new List<Point2dFloat>(components.Count());
            compMedians = new List<float>(components.Count());
            compDimensions = new List<Point2d>(components.Count());

            compBB = new List<Tuple<Point2d, Point2d>>(components.Count());

            foreach (List<Point2d> component in components)
            {
                if (component.Count() > 0)
                {
                    float mean = 0, variance = 0, median = 0;
                    int minx = 0, miny = 0, maxx = 0, maxy = 0;


                    componentStats(SWTImage, component, ref mean, ref variance, ref median, ref minx, ref miny, ref maxx, ref maxy);

                    // check if variance is less than half the mean
                    if (Math.Sqrt(variance) > 0.5 * mean)
                    {
                        continue;
                    }

                    float width = (float)(maxx - minx + 1);
					float height = ( float )( maxy - miny + 1 );

                    // check font height too big (normal characters are 80 pixels high, for acA1300)
					if( height > 300 )
                    {
                        continue;
                    }

					// check font height too small
					if( height <  50)
					{
						continue;
					}

					float area = width * height;
                    float rminx = (float)minx;
                    float rmaxx = (float)maxx;
                    float rminy = (float)miny;
                    float rmaxy = (float)maxy;
                    // compute the rotated bounding box
                    float increment = 1.0f / 36.0f;

                    for (float theta = increment * (float)Math.PI; theta < Math.PI / 2.0f; theta += increment * (float)Math.PI)
                    {
                        float xmin, xmax, ymin, ymax, xtemp, ytemp, ltemp, wtemp;
                        xmin = 1000000;
                        ymin = 1000000;
                        xmax = 0;
                        ymax = 0;
                        for (int i = 0; i < component.Count(); i++)
                        {
                            xtemp = (component)[i].x * (float)Math.Cos(theta) + (component)[i].y * -(float)Math.Sin(theta);
                            ytemp = (component)[i].x * (float)Math.Sin(theta) + (component)[i].y * (float)Math.Cos(theta);
                            xmin = Math.Min(xtemp, xmin);
                            xmax = Math.Max(xtemp, xmax);
                            ymin = Math.Min(ytemp, ymin);
                            ymax = Math.Max(ytemp, ymax);
                        }
                        ltemp = xmax - xmin + 1;
                        wtemp = ymax - ymin + 1;
                        if (ltemp * wtemp < area)
                        {
                            area = ltemp * wtemp;
							width = ltemp;
							height = wtemp;
                        }
                    }

                    // check if the aspect ratio is between 1/10 and 10
					if( width / height < 1.0f / 10.0f || width / height > 10.0 )
                    {
                        continue;
                    }

                    // compute the diameter TODO finish
                    // compute dense representation of component
                    List<List<float>> denseRepr = new List<List<float>>(maxx - minx + 1);
                    for (int i = 0; i < maxx - minx + 1; i++)
                    {
                        List<float> tmp = new List<float>(maxy - miny + 1);
                        denseRepr.Add(tmp);
                        for (int j = 0; j < maxy - miny + 1; j++)
                        {
                            denseRepr[i].Add(0);
                        }
                    }
                    foreach (Point2d pit in component)
                    {
                        (denseRepr[pit.x - minx])[pit.y - miny] = 1;
                    }
                    // create graph representing components
                    int num_nodes = component.Count();
                    /*
                    E edges[] = { E(0,2),
                                  E(1,1), E(1,3), E(1,4),
                                  E(2,1), E(2,3),
                                  E(3,4),
                                  E(4,0), E(4,1) };

                    Graph G(edges + sizeof(edges) / sizeof(E), weights, num_nodes);
                    */
                    Point2dFloat center;
                    center.x = ((float)(maxx + minx)) / 2.0f;
                    center.y = ((float)(maxy + miny)) / 2.0f;

                    Point2d dimensions = new Point2d();
                    dimensions.x = maxx - minx + 1;
                    dimensions.y = maxy - miny + 1;

                    Point2d bb1 = new Point2d();
                    bb1.x = minx;
                    bb1.y = miny;

                    Point2d bb2 = new Point2d();
                    bb2.x = maxx;
                    bb2.y = maxy;
                    Tuple<Point2d, Point2d> pair = new Tuple<Point2d, Point2d>(bb1, bb2);

                    compBB.Add(pair);
                    compDimensions.Add(dimensions);
                    compMedians.Add(median);
                    compCenters.Add(center);
                    validComponents.Add(component);
                }
            }
            List<List<Point2d >> tempComp = new List<List<Point2d>>(validComponents.Count());
            List<Point2d > tempDim = new List<Point2d>(validComponents.Count());
            List<float > tempMed = new List<float>(validComponents.Count());
            List<Point2dFloat > tempCenters = new List<Point2dFloat>((validComponents.Count()));
            List<Tuple<Point2d,Point2d> > tempBB = new List<Tuple<Point2d,Point2d>>(validComponents.Count());

            for (int i = 0; i < validComponents.Count(); i++) 
            {
                int count = 0;
                for (int j = 0; j < validComponents.Count(); j++)
                {
                    if (i != j)
                    {
                        // component center of component is inside another component
                        if (compBB[i].Item1.x <= compCenters[j].x && compBB[i].Item2.x >= compCenters[j].x &&
                            compBB[i].Item1.y <= compCenters[j].y && compBB[i].Item2.y >= compCenters[j].y) 
                        {
                            count++;
                        }
                    }
                }
                if (count < 2) 
                {// component is unique
                    tempComp.Add(validComponents[i]);
                    tempCenters.Add(compCenters[i]);
                    tempMed.Add(compMedians[i]);
                    tempDim.Add(compDimensions[i]);
                    tempBB.Add(compBB[i]);
                }
            }
            
            validComponents = tempComp;
            compDimensions = tempDim;
            compMedians = tempMed;
            compCenters = tempCenters;
            compBB = tempBB;

        }
    
    
        public static void componentStats(IplImage SWTImage,
                                        List<Point2d> component,
                                        ref float mean, ref float variance, ref float median,
                                        ref int minx, ref int miny, ref int maxx, ref int maxy)
        {
            unsafe
            {
                float* swtPtr = (float*)SWTImage.ImageData.ToPointer();
                int swtWidthStep = SWTImage.WidthStep / 4;

                List<float> temp = new List<float>(component.Count());
                mean = 0;
				double varDouble = 0;
                variance = 0;
                minx = 1000000;
                miny = 1000000;
                maxx = 0;
                maxy = 0;

                foreach (Point2d it in component)
                {
                    float t = swtPtr[it.y * swtWidthStep + it.x];
					if( t > 0 )
					{
						mean += t;
						temp.Add( t );
					}
					miny = Math.Min( miny, it.y );
					minx = Math.Min( minx, it.x );
					maxy = Math.Max( maxy, it.y );
					maxx = Math.Max( maxx, it.x );
					
                }
                mean = mean / ((float)temp.Count());

                foreach (float it in temp)
                {

					varDouble += (double)( it - mean ) * (double)( it - mean );
                }
                variance = (float)varDouble / ((float)temp.Count());
                temp.Sort();

                median = temp[temp.Count() / 2];
            }
        }
        public static void FilterRays(IplImage SWTImage, List<Ray> rays, IplImage cleanSWTImage)
		{
			// get stats on rays
			Rays raySet = new Rays(rays);
			raySet.SetValid( true );

			cleanSWTImage.Set(-1);
			unsafe
			{
				float* swtPtr = ( float* )SWTImage.ImageData.ToPointer();
				int swtWidthStep = SWTImage.WidthStep / 4;
				float* cleanSwtPtr = ( float* )cleanSWTImage.ImageData.ToPointer();
				int cleanSwtWidthStep = SWTImage.WidthStep / 4;
				// filter rays that are not right length
				for(int i = 0; i< raySet.rayLengths.Count();i++ )
				{
					if( raySet.rayLengths[i] < raySet.median - 0.5*raySet.stdev || raySet.rayLengths[i] > raySet.median + 0.5*raySet.stdev )
					{
						raySet.rays[ i ].valid = false;
					}
				}


				for( int i = 0; i < raySet.rayLengths.Count(); i++ )
				{
					if (raySet.rays[i].valid)
					{
						List<Point2d> points = raySet.rays[i].points;
						for( int j = 0; j < points.Count();j++ )
							cleanSwtPtr[ points[ j ].y * swtWidthStep + points[ j ].x ] = swtPtr[ points[ j ].y * swtWidthStep + points[ j ].x ];

					}
				}
				
			}

		}

		static List<List<Point2d>> FindLegallyConnectedComponentsRAY( IplImage SWTImage, List<Ray> rays )
		{
			Dictionary<int, int> map = new Dictionary<int, int>();
			Dictionary<int, Point2d> revMap = new Dictionary<int, Point2d>();

			return null;
		}


		
	public static List<List<Point2d>> FindLegallyConnectedComponents(IplImage SWTImage, List<Ray> rays)
		{
			Dictionary<int, int> map = new Dictionary<int, int>();
			
			Dictionary<int, Point2d> revMap = new Dictionary<int, Point2d>();

			int numVertices = 0;

			unsafe
			{
                // adding each point to map of points and an index
				float* swtPtr = ( float* )SWTImage.ImageData.ToPointer();
				int SWTWidthStep = SWTImage.WidthStep/4;

				for( int row = 0; row < SWTImage.Height; row++ )
				{
					for( int col = 0; col < SWTImage.Width; col++ )
					{
						if( swtPtr[row*SWTWidthStep+col] > 0 )
						{
							map.Add(row * SWTImage.Width + col, numVertices);
							Point2d p = new Point2d();
							p.x = col;
							p.y = row;
							revMap.Add( numVertices, p);
							numVertices++;
						}
					}
				}


				Graph graph = new Graph();// key is vertex, list of connected vertices, vertex value is current set vertex value, visited?

                // why connected only right, down, downright, downleft
				for( int row = 0; row < SWTImage.Height; row++ )
				{
					for (int col = 0; col < SWTImage.Width; col++ )
					{
						float value = swtPtr[row*SWTWidthStep+col];
						if (value > 0)
						{
							// check pixel to the right, right-down, down, left-down
							int this_pixel = map[row * SWTImage.Width + col];
                            graph.Add(this_pixel);

							if( row - 1 >= 0 )
							{// up
								if( col + 1 < SWTImage.Width )
								{// right up
									float right_up = swtPtr[ ( row - 1 ) * SWTWidthStep + col + 1 ];
									if( right_up > 0 && ( ( value ) / right_up <= 3.0 || right_up / ( value ) <= 3.0 ) )
									{
										graph.Add( this_pixel, map[ ( row - 1 ) * SWTImage.Width + col + 1 ] );
									}
								}
								float up = swtPtr[ ( row - 1 ) * SWTWidthStep + col ];
								if( up > 0 && ( ( value ) / up <= 3.0 || up / ( value ) <= 3.0 ) )
								{
									graph.Add( this_pixel, map[ ( row - 1 ) * SWTImage.Width + col ] );
								}
								if( col - 1 >= 0 )
								{// up left
									float left_up = swtPtr[ ( row - 1 ) * SWTWidthStep + col - 1 ];
									if( left_up > 0 && ( ( value ) / left_up <= 3.0 || left_up / ( value ) <= 3.0 ) )
									{
										graph.Add( this_pixel, map[ ( row - 1 ) * SWTImage.Width + col - 1 ] );
									}
								}
							}
							// middle
							if (col+1 < SWTImage.Width) {//right
								float right = swtPtr[row*SWTWidthStep+col+1];
								if (right > 0 && ((value)/right <= 3.0 || right/(value) <= 3.0))
								{
									graph.Add(this_pixel, map[row*SWTImage.Width+col+1]);
								}
							}
							if( col - 1 >=0 )
							{//right
								float left = swtPtr[ row * SWTWidthStep + col - 1 ];
								if( left > 0 && ( ( value ) / left <= 3.0 || left / ( value ) <= 3.0 ) )
								{
									graph.Add( this_pixel, map[ row * SWTImage.Width + col - 1 ] );
								}
							}
							// down
							if (row+1 < SWTImage.Height)
							{
								if (col+1 < SWTImage.Width)
								{//right down
									float right_down = swtPtr[(row+1)*SWTWidthStep+col+1];
									if (right_down > 0 && ((value)/right_down <= 3.0 || right_down/(value) <= 3.0))
									{
										graph.Add(this_pixel, map[(row+1)*SWTImage.Width+col+1]);
									}
								}
								float down = swtPtr[(row+1)*SWTWidthStep+col];
								if (down > 0 && ((value)/down <= 3.0 || down/(value) <= 3.0))
								{//down
									graph.Add(this_pixel, map[(row+1)*SWTImage.Width+col]);
								}
								if (col-1 >= 0)
								{// left down
									float left_down = swtPtr[(row+1)*SWTWidthStep+col-1];
									if( left_down > 0 && ( ( value ) / left_down <= 3.0 || left_down / ( value ) <= 3.0 ) )
									{
										graph.Add(this_pixel, map[(row+1)*SWTImage.Width+col-1]);
									}
								}
							}
						}
					}
				}

				// 
				List<int> c = new List<int>();
                int numComp = connectedComponentsBFS(graph, ref c);

                List<List<Point2d>> components = new List<List<Point2d>>(numComp);
                components.Add(new List<Point2d>());
                for (int j = 0; j < numComp; j++)
                {
                    components.Add(new List<Point2d>());
                }

                for (int j = 0; j < numVertices; j++)
                {
                    Point2d p = revMap[j];
                    components[c[j]].Add(p);
                }

                return components;

			}

            
		}

        public static int connectedComponents(Graph graph, ref List<int> components)
        {// key is vertex, list of connected vertices, vertex value is current set vertex value, visited?
            
            // reset graph visited
            graph.ResetVisited();

			// do until all components of graphs have been visited
            GraphNode graphNode = graph.NextUnvisited();
            int vertexNumber = 1;
            while (graphNode != null)
            {
                graphNode.visited = true;
                
                DepthFirstSearch(ref graph, graphNode.vertexValue, vertexNumber);
                graphNode.vertexValue = vertexNumber;
                vertexNumber++;
                graphNode = graph.NextUnvisited();
            }

            // as many components as vertices --> component contains the components that the edge point belongs to
            components = new List<int>();
            foreach (KeyValuePair<int, GraphNode> graphnode in graph)
            {
                components.Add(graphnode.Value.vertexValue);
            }

            // return the number of different components
            return vertexNumber - 1;
        }

		public static int connectedComponentsBFS( Graph graph, ref List<int> components )
		{// key is vertex, list of connected vertices, vertex value is current set vertex value, visited?

			// reset graph visited
			graph.ResetVisited();

			// do until all components of graphs have been visited
			GraphNode graphNode = graph.NextUnvisited();
			int vertexNumber = 1;
			while( graphNode != null )
			{
				graphNode.visited = true;

				BreadthFirstSearch( ref graph, graphNode.vertexValue, vertexNumber );
				graphNode.vertexValue = vertexNumber;
				vertexNumber++;
				graphNode = graph.NextUnvisited();
			}

			// as many components as vertices --> component contains the components that the edge point belongs to
			components = new List<int>();
			foreach( KeyValuePair<int, GraphNode> graphnode in graph )
			{
				components.Add( graphnode.Value.vertexValue );
			}

			// return the number of different components
			return vertexNumber - 1;
		}

        public static void BreadthFirstSearch(ref Graph graph, int key, int vertexNumber)
		{
			Stack<GraphNode> connComp = new Stack<GraphNode>();
			connComp.Push( graph[ key ] );
			
			while (connComp.Count()>0)
			{
				GraphNode gn = connComp.Pop();
				gn.visited = true;
				gn.vertexValue = vertexNumber;
				for (int i = 0;i<gn.adjacency.Count();i++)
				{
					if (graph[gn.adjacency[i]].visited == false)
					{
						connComp.Push( graph[ gn.adjacency[ i ] ] );
					}
				}
			}

		}

        public static void DepthFirstSearch(ref Graph graph, int key, int vertexNumber)
        {
            if (vertexNumber == 2 && key==10667)
			{

			}
            List<int> nodes = graph[key].adjacency;
            for (int i = 0; i < nodes.Count(); i++)
            {
                if (graph[nodes[i]].visited == false)
                {
                    graph[nodes[i]].vertexValue = vertexNumber;
                    graph[nodes[i]].visited = true;
                    DepthFirstSearch(ref graph, nodes[i], vertexNumber);
                }
            }

            
        }


		public static void SWTMedianFilter (IplImage SWTImage, List<Ray> rays)
		{
			unsafe
			{
				float* swtPtr = ( float* )SWTImage.ImageData.ToPointer();
				int SWTWidthStep = SWTImage.WidthStep/4;
			
				Point2dComparer p2dComp = new Point2dComparer();

				foreach (Ray ray in rays)
				{
					for( int i = 0; i < ray.points.Count();i++ )
					{
						Point2d pt = ray.points[ i ];
						pt.SWT = swtPtr[ pt.y * SWTWidthStep + pt.x ];
						ray.points[ i ] = pt;
					}


					ray.points.Sort(p2dComp);

					float median = ray.points[ray.points.Count()/2].SWT;
					foreach (Point2d point in ray.points)
					{
						swtPtr[point.y * SWTWidthStep + point.x] = Math.Min(point.SWT,median);
					}
				}
			}
			
		}




		

		public static void strokeWidthTransform( IplImage edgeImage, IplImage gradientX, IplImage gradientY, bool darkOnLight, IplImage SWTImage, List<Ray> rays )
		{
			unsafe// looks safe
			{
				float prec = 0.05f;
				byte* img1 = ( byte* )edgeImage.ImageData.ToPointer();
				int srcWidthStep = edgeImage.WidthStep;

				float* gradX = ( float* )gradientX.ImageData.ToPointer();
				int gradXWidthStep = gradientX.WidthStep/4;

				float* gradY = ( float* )gradientY.ImageData.ToPointer();
				int gradYWidthStep = gradientY.WidthStep/4;

				float* swtPtr = ( float* )SWTImage.ImageData.ToPointer();
				int SWTWidthStep = SWTImage.WidthStep/4;

				for (int row = 0;row<edgeImage.Height;row++)
				{
					for (int col = 0;col<edgeImage.Width;col++)
					{
						if (img1[row*srcWidthStep + col] > 0)
						{
							Ray r = new Ray();

							Point2d p = new Point2d() ;
							p.x = col;
							p.y = row;
							r.p = p;
							List<Point2d> points = new List<Point2d>();
							points.Add( p );

							float curX = ( float )col + 0.5f;
							float curY = ( float )row + 0.5f;
							int curPixX = col;
							int curPixY = row;
							float Gx = gradX[ row * gradXWidthStep + col ];
							float Gy = gradY[ row * gradYWidthStep + col ];
							// normalize gradient
							float mag = (float)Math.Sqrt( Gx * Gx + Gy * Gy );
							if( darkOnLight )
							{
								Gx = -Gx / mag;
								Gy = -Gy / mag;
							}
							else
							{
								Gx = Gx / mag;
								Gy = Gy / mag;
							}

							while( true )
							{
								curX += Gx * prec;
								curY += Gy * prec;

								if( ( int )( Math.Floor( curX ) ) != curPixX || ( int )( Math.Floor( curY ) ) != curPixY )
								{
									curPixX = ( int )( Math.Floor( curX ) );
									curPixY = ( int )( Math.Floor( curY ) );
									// check if pixel is outside boundary of image
									if( curPixX < 0 || ( curPixX >= SWTImage.Width ) || curPixY < 0 || ( curPixY >= SWTImage.Height ) )
									{
										break;
									}
									Point2d pnew = new Point2d();
									pnew.x = curPixX;
									pnew.y = curPixY;
									points.Add( pnew );

									if( img1[ curPixY * srcWidthStep + curPixX ] > 0 )
									{
										r.q = pnew;
										float Gxt = gradX[ curPixY * gradXWidthStep + curPixX ];
										float Gyt = gradY[ curPixY * gradYWidthStep + curPixX ];

										mag = ( float )Math.Sqrt( Gxt * Gxt + Gyt * Gyt );
										if( darkOnLight )
										{
											Gxt = -Gxt / mag;
											Gyt = -Gyt / mag;
										}
										else
										{
											Gxt = Gxt / mag;
											Gyt = Gyt / mag;
										}

										if( Math.Acos( Gx * -Gxt + Gy * -Gyt ) < Math.PI / 2.0 )
										{
											float length = ( float )Math.Sqrt( ( ( float )r.q.x - ( float )r.p.x ) * ( ( float )r.q.x - ( float )r.p.x ) + ( ( float )r.q.y - ( float )r.p.y ) * ( ( float )r.q.y - ( float )r.p.y ) );
											foreach( Point2d point in points )
											{
												if( swtPtr[ point.y * SWTWidthStep + point.x ] < 0 )
												{
													swtPtr[ point.y * SWTWidthStep + point.x ] = length;
												}
												else
												{
													swtPtr[ point.y * SWTWidthStep + point.x ] = Math.Min( length, swtPtr[ point.y * SWTWidthStep + point.x ] );
												}
											}
											r.points = points;
											rays.Add( r );

										}
										break;
									}
								}
							}
						}
					}
				}

			}
		}

	}
}
