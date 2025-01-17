package Project;

import java.awt.*;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.Function;
import java.io.File;



import ij.IJ;
import ij.ImagePlus;
import ij.process.ColorProcessor;
import ij.process.FloatProcessor;
import ij.process.FloodFiller;
import ij.process.ImageProcessor;

public class DressFeatureExtractor
{

    public static void main(String[] args) {

        String baseFolder = "src/main/resources/metaqse/";
        Map<String, String> labelMap = new HashMap<>();
        labelMap.put("FLORAL_IMAGES", "patterned");
        labelMap.put("POLKADOT_IMAGES", "patterned");
        labelMap.put("STRIPED_IMAGES", "patterned");
        labelMap.put("SOLID_IMAGES", "solid");

        String outputCsv = "src/main/resources/metaqse/output/features_binary.csv";

        // Prepare CSV header
        List<String> headers = Arrays.asList(
            "Image_Name", "Label",
            "Dominant_Color_Percentage", "GradX_Avg", "GradY_Avg",
            "Homogeneity_0", "Energy_0", "Contrast_0", "Correlation_0", "Entropy_0"
        );

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputCsv))) {
            // Write the CSV header
            writer.write(String.join(",", headers));
            writer.newLine();

            // Process each folder
            for (Map.Entry<String, String> entry : labelMap.entrySet()) {
                String folderName = entry.getKey();
                String label = entry.getValue();

                File folder = new File(baseFolder + folderName);
                File[] imageFiles = folder.listFiles((dir, name) -> name.toLowerCase().endsWith(".jpg") ||
                    name.toLowerCase().endsWith(".png") || name.toLowerCase().endsWith(".jpeg"));

                if (imageFiles != null) {
                    for (File imageFile : imageFiles) {
                        System.out.println("Processing: " + imageFile.getName());
                        Map<String, Object> features = extractFeatures(imageFile.getAbsolutePath());
                        features.put("Image_Name", imageFile.getName());
                        features.put("Label", label);

                        // Write to CSV
                        writeCsvRow(writer, features, headers);
                    }
                }
            }

            System.out.println("Feature extraction completed. Results saved to: " + outputCsv);

        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private static Map<String, Object> extractFeatures(String imagePath) {
        Map<String, Object> finalResultFeatures = new LinkedHashMap<>();

        ImagePlus imp = IJ.openImage(imagePath);
        imp = imp.resize(300, 700, "bicubic");


        final List<HSV> firstLeftHSVColors = hsvColors((ColorProcessor) imp.getProcessor(), 2, 10);
        final List<HSV> lastRightHSVColors = hsvColors((ColorProcessor) imp.getProcessor(), imp.getProcessor().getHeight() - 6, 10);

        Canny_Edge_Detector canny = new Canny_Edge_Detector();
        // Set parameters
        canny.setLowThreshold(1f);
        canny.setHighThreshold(3f);
        canny.setGaussianKernelRadius(3f);
        canny.setGaussianKernelWidth(16);
        canny.setContrastNormalized(false);

        // Process the image
        ImagePlus edges = canny.process(imp);
        ImagePlus dup = edges.duplicate();
        dup.setTitle("Edges processed");

        for (int i = 0; i < 20; i++) {
            edges.getProcessor().erode();
        }

        ImagePlus isolatedObject = new ImagePlus("Errosion Object", edges.getProcessor().duplicate());

        FloodFiller floodFiller = new FloodFiller(edges.getProcessor());
        int seedX = edges.getWidth() / 2;
        int seedY = edges.getHeight() / 2;
        edges.getProcessor().setColor(255);
        floodFiller.fill(seedX, seedY);
        ImagePlus imagePlus = new ImagePlus("FloodFill", edges.getProcessor().duplicate());

        ImageProcessor originalProcessor = imp.getProcessor();
        ImageProcessor resultProcessor = originalProcessor.duplicate();
        // Apply the mask
        for (int y = 0; y < originalProcessor.getHeight(); y++) {
            for (int x = 0; x < originalProcessor.getWidth(); x++) {
                if (edges.getProcessor().getPixel(x, y) == 0) {
                    resultProcessor.putPixel(x, y, 0xffffff); // Set the pixel to white (or any background color)
                }
            }
        }



        // Create a new ImagePlus to hold the result
        ImagePlus resultImage = new ImagePlus(imagePath, resultProcessor);
        // Process the image to detect skin
        int[] pixels = (int[]) resultImage.getProcessor().getPixels();
        float[] hsb = new float[3];

        //
        Map<Integer, List<HSV>> hueStatics = new TreeMap<>();
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            int red = (pixel & 0xff0000) >> 16;
            int green = (pixel & 0x00ff00) >> 8;
            int blue = (pixel & 0x0000ff);
            Color.RGBtoHSB(red, green, blue, hsb);

            // Use hsb[0], hsb[1], and hsb[2] for hue, saturation, and brightness respectively
            // to detect skin pixels and perform further processing...
            float hue = hsb[0] * 360; // Convert hue to degrees
            float saturation = hsb[1];
            float brightness = hsb[2];

            // Define skin color thresholds
            // These will need fine-tuning for your specific application
            float hueLowerThreshold = 0; // lower hue in degrees
            float hueUpperThreshold = 50; // upper hue in degrees
            float saturationLowerThreshold = 0.2f;
            float saturationUpperThreshold = 0.7f;

            float averageAboveBackgroundHue = calculateAverageFeature(firstLeftHSVColors, HSV::getHue);
            float averageLowerBackgroundHue = calculateAverageFeature(lastRightHSVColors, HSV::getHue);

            float averageAboveBackgroundSat = calculateAverageFeature(firstLeftHSVColors, HSV::getSaturation);
            float averageLowerBackgroundSat = calculateAverageFeature(lastRightHSVColors, HSV::getSaturation);

            float averageAboveBackgroundVal = calculateAverageFeature(firstLeftHSVColors, HSV::getValue);
            float averageLowerBackgroundVal = calculateAverageFeature(lastRightHSVColors, HSV::getValue);


            // Check if the pixel falls within the skin color HSV range
            if (hue >= hueLowerThreshold && hue <= hueUpperThreshold &&
                saturation >= saturationLowerThreshold && saturation <= saturationUpperThreshold) {
                pixels[i] = 0xffffff; // White color to mask the skin
            }

            double distance1  = calculateDistance(hsb[0], saturation, brightness, averageAboveBackgroundHue/360f, averageAboveBackgroundSat, averageAboveBackgroundVal);
            double distance2  = calculateDistance(hsb[0], saturation, brightness, averageLowerBackgroundHue/360f, averageLowerBackgroundSat, averageLowerBackgroundVal);

            if(distance1 < 0.1f  || distance2 < 0.1f) {
                pixels[i] = 0xffffff;
            }

            pixel = pixels[i];
            red = (pixel & 0xff0000) >> 16;
            green = (pixel & 0x00ff00) >> 8;
            blue = (pixel & 0x0000ff);
            Color.RGBtoHSB(red, green, blue, hsb);

            // Use hsb[0], hsb[1], and hsb[2] for hue, saturation, and brightness respectively
            // to detect skin pixels and perform further processing...
            hue = hsb[0] * 360; // Convert hue to degrees
            saturation = hsb[1];
            brightness = hsb[2];
            int bucketNumber  = (int)hue/30;
            if(hue!=0 || saturation!=0 || brightness!=1) {
                if (hueStatics.get(bucketNumber) == null) {
                    List<HSV> hsvs = new ArrayList<>();
                    hsvs.add(new HSV(hue, saturation, brightness));
                    hueStatics.put(bucketNumber, hsvs);
                } else {
                    List<HSV> hsvs = hueStatics.get(bucketNumber);
                    hsvs.add(new HSV(hue, saturation, brightness));
                    hueStatics.put(bucketNumber, hsvs);
                }
            }
        }
        int maxCount = 0;
        int maxSection = 0;
        int totalBucketSize  = 0;
        for(int i=0; i< 24; i++) {
            if(hueStatics.get(i) != null) {
                totalBucketSize += hueStatics.get(i).size();
                List<HSV> hsvs = hueStatics.get(i);
                if(hsvs.size() >maxCount) {
                    maxCount = hsvs.size();
                    maxSection = i;
                }
            }
        }

        float percentage  = (float)maxCount/totalBucketSize;

        FloatProcessor hueProcessor  = new FloatProcessor(resultProcessor.getWidth(), resultProcessor.getHeight());
        for(int y = 0; y<resultProcessor.getHeight(); y++) {
            for(int x =0; x<resultProcessor.getWidth(); x++) {
                int rgb = resultProcessor.get(x,y);
                int red = (rgb & 0xff0000) >> 16;
                int green = (rgb & 0x00ff00) >> 8;
                int blue = (rgb & 0x0000ff);
                Color.RGBtoHSB(red, green, blue, hsb);
                hueProcessor.putPixelValue(x, y, hsb[0] * 360);
            }
        }

        ImageProcessor gradientXImageProcessor = hueProcessor.duplicate();
        ImageProcessor gradientYImageProcessor = hueProcessor.duplicate();

        gradientXImageProcessor.convolve(new float[] {-1, 0, 1, -2, 0, 2, -1, 0, 1}, 3, 3);
        gradientYImageProcessor.convolve(new float[] {-1, -2, -1, 0, 0, 0, 1, 2, 1}, 3, 3);


        Map<String, List<Float>> gradientProcessorSumAndCountPerOrientation = new HashMap<>();

        for(int i = 0; i<gradientXImageProcessor.getHeight(); i++) {
            for(int j=0; j<gradientXImageProcessor.getWidth(); j++) {
                if(gradientXImageProcessor.getPixelValue(j,i) != 0) {
                    computeAverageAndSum(
                        gradientProcessorSumAndCountPerOrientation,
                        gradientXImageProcessor,
                        j,
                        i,
                        "x");
                }
                if(gradientYImageProcessor.getPixelValue(j,i) != 0) {
                    computeAverageAndSum(
                        gradientProcessorSumAndCountPerOrientation,
                        gradientYImageProcessor,
                        j,
                        i,
                        "y"
                    );
                }
            }
        }

        ImageProcessor graysScaleImage  = resultProcessor.duplicate();
        graysScaleImage.convertToByteProcessor();

        double[][] glcm = GLCMCalculator.calculateGLCM(graysScaleImage, 1, "0");

        Map<String, Double> glcmFeatures  = GLCMCalculator.calculateFeatures(glcm);

        finalResultFeatures.put("Dominant_Color_Percentage", percentage);
        finalResultFeatures.put("GradX_Avg", gradientProcessorSumAndCountPerOrientation.get("x").get(1)/gradientProcessorSumAndCountPerOrientation.get("x").get(0));
        finalResultFeatures.put("GradY_Avg",  gradientProcessorSumAndCountPerOrientation.get("y").get(1)/gradientProcessorSumAndCountPerOrientation.get("y").get(0));
        finalResultFeatures.put("Homogeneity_0", glcmFeatures.get("Homogeneity"));
        finalResultFeatures.put("Energy_0", glcmFeatures.get("Energy"));
        finalResultFeatures.put("Contrast_0", glcmFeatures.get("Contrast"));
        finalResultFeatures.put("Correlation_0", glcmFeatures.get("Correlation"));
        finalResultFeatures.put("Entropy_0", glcmFeatures.get("Entropy"));

        System.out.println("The dominant color percentage is " + percentage);
        System.out.println("Average gradX value is " +
            gradientProcessorSumAndCountPerOrientation.get("x").get(1)/gradientProcessorSumAndCountPerOrientation.get("x").get(0));
        System.out.println("Average gradY value is " +
            gradientProcessorSumAndCountPerOrientation.get("y").get(1)/gradientProcessorSumAndCountPerOrientation.get("y").get(0));

        System.out.println("Homogeneity 0 deg is " + glcmFeatures.get("Homogeneity"));
        System.out.println("Energy 0 deg is " + glcmFeatures.get("Energy"));
        System.out.println("Contrast 0 deg is " + glcmFeatures.get("Contrast"));
        System.out.println("Correlation 0 deg is " + glcmFeatures.get("Correlation"));
        System.out.println("Entropy 0 deg is " + glcmFeatures.get("Entropy"));
        resultImage.show();
        resultImage.show();

        return finalResultFeatures;
    }

    private static void computeAverageAndSum(Map<String, List<Float>> gradientProcessorSumAndCountPerOrientation,
                                             ImageProcessor imageProcessor,
                                             final int x,
                                             final int y,
                                             String typeOfProcessor
                                             ){
        gradientProcessorSumAndCountPerOrientation.compute(typeOfProcessor, (k, v) -> {
            if(v == null) {
                ArrayList<Float> sumAndCount = new ArrayList<>();
                sumAndCount.add(1f);
                sumAndCount.add(imageProcessor.getPixelValue(x, y));
                return sumAndCount;
            } else {
                float sum = gradientProcessorSumAndCountPerOrientation.get(typeOfProcessor).get(1) + imageProcessor.getPixelValue(x, y);
                float totalNumber  = gradientProcessorSumAndCountPerOrientation.get(typeOfProcessor).get(0) + 1;
                v.set(0, totalNumber);
                v.set(1, sum);
                return v;
            }
        });
    }
    private static List<HSV> hsvColors(ColorProcessor cp, int startY, int edgeWidth) {
        int[] RGB = new int[3];
        List<HSV> hsvs = new ArrayList<>();

            for (int x = 0; x < edgeWidth; x++) {
                cp.getPixel(x, startY, RGB);
                float[] arrHSV = Color.RGBtoHSB(RGB[0], RGB[1], RGB[2], null);
                HSV hsv = new HSV(arrHSV[0] * 360, arrHSV[1], arrHSV[2]);
                hsvs.add(hsv);
            }

        return hsvs;
    }

    private static double calculateDistance(float h1, float s1, float v1, float h2, float s2, float v2) {
        return Math.sqrt(
                Math.pow((v2 - v1), 2) +
                        Math.pow(s1 * v1, 2) + Math.pow(s2 * v2, 2) -
                        2 * s1 * s2 * v1 * v2 * Math.cos((h2 - h1) * Math.PI) // Corrected to use radians
        );
    }

    private static float calculateAverageFeature(List<HSV> hsvs, Function<HSV, Float> hsvConsumer) {
        float sum = 0;
        for(int i = 0; i<hsvs.size(); i++) {
            sum += hsvConsumer.apply(hsvs.get(i));
        }
        return sum/hsvs.size();
    }

    private static void writeCsvRow(BufferedWriter writer, Map<String, Object> row, List<String> headers) throws IOException {
        List<String> values = new ArrayList<>();
        for (String header : headers) {
            values.add(row.getOrDefault(header, "").toString());
        }
        writer.write(String.join(",", values));
        writer.newLine();
    }

}
