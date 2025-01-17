package Project;

import java.util.HashMap;
import java.util.Map;

import ij.process.ImageProcessor;

public class GLCMCalculator
{


    public static double[][] calculateGLCM(ImageProcessor imageProcessor, int step, String direction) {
        if(!(direction.equals("0") || direction.equals("90"))) {
            throw new IllegalArgumentException("The direction should be either 0 or 90 degrees");
        }

        int width  = imageProcessor.getWidth();
        int height  = imageProcessor.getHeight();

        double[][] glcm = new double[257][257];
        int occurrenceCount  = 0;

        if(direction.equals("0")) {
            for(int y=0; y<height; y++) {
                for (int x=0; x<width; x++) {
                    int a = (int) imageProcessor.getPixelValue(x, y);
                    int b = (int) imageProcessor.getPixelValue(x + step, y);

                    occurrenceCount++;
                    glcm[a][b] += 1;
                    glcm[b][a] += 1;
                }
            }
        } else {
            for(int y=0; y<height; y++) {
                for (int x=0; x<width; x++) {
                    int a = (int) imageProcessor.getPixelValue(x, y);
                    int b = (int) imageProcessor.getPixelValue(x, y + step);

                    occurrenceCount++;
                    glcm[a][b] += 1;
                    glcm[b][a] += 1;
                }
            }
        }

        if(occurrenceCount == 0) {
            throw new IllegalStateException("The occurrence count can't be zero");
        }

        for (int y = 0; y < 256; y++) {
            for (int x = 0; x < 256; x++) {
                glcm[x][y] /= occurrenceCount;
            }
        }

        return glcm;
    }

    public static Map<String, Double> calculateFeatures(double[][] glcm) {
        double contrast = 0.0;
        double energy = 0.0;
        double homogeneity = 0.0;
        double correlation = 0.0;
        double entropy = 0.0;

        double meanRow = 0.0;
        double meanCol = 0.0;
        double stdDevRow = 0.0;
        double stdDevCol = 0.0;

        for (int i = 0; i < glcm.length; i++) {
            for (int j = 0; j < glcm[0].length; j++) {
                meanRow += i * glcm[i][j];
                meanCol += j * glcm[i][j];
            }
        }
        for (int i = 0; i < glcm.length; i++) {
            for (int j = 0; j < glcm[0].length; j++) {
                stdDevRow += Math.pow(i - meanRow, 2) * glcm[i][j];
                stdDevCol += Math.pow(j - meanCol, 2) * glcm[i][j];
            }
        }
        stdDevRow = Math.sqrt(stdDevRow);
        stdDevCol = Math.sqrt(stdDevCol);

        // Calculate features
        for (int i = 0; i < glcm.length; i++) {
            for (int j = 0; j < glcm[0].length; j++) {
                double p = glcm[i][j];

                // Contrast
                contrast += (i - j) * (i - j) * p;

                // Energy
                energy += p * p;

                // Homogeneity
                homogeneity += p / (1 + Math.abs(i - j));

                // Entropy
                if (p > 0) {
                    entropy -= p * Math.log(p);
                }

                // Correlation
                correlation += ((i - meanRow) * (j - meanCol) * p) / (stdDevRow * stdDevCol);
            }
        }
        Map<String, Double> features = new HashMap<>();
        features.put("Contrast", contrast);
        features.put("Energy", energy);
        features.put("Homogeneity", homogeneity);
        features.put("Correlation", correlation);
        features.put("Entropy", entropy);

        return features;
    }

}
