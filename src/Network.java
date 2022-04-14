import java.io.*;
import java.util.Arrays;
import java.util.Scanner;

/*
 * This class represents an A by B by C by D perceptron. There are A neurons in the input layer,
 * B neurons in the first hidden layer, and C neurons in the second hidden layer, and D neurons in the output layer.
 *
 * The network takes in sets of input test cases and can train itself with backpropagation to predict the outputs of
 * those test cases. Floating point weights connect the neurons between each layer of the network.
 *
 * @author Anirudh Kotamraju
 * @version April 13, 2022
 */
public class Network
{
   /*
    * Static variables.
    */

   public static double[][] firstLayerWeights;  // Weights connecting the input and first hidden layer.
   public static double[][] secondLayerWeights; // Weights connecting the first hidden and second hidden layer.
   public static double[][] thirdLayerWeights;  // Weights connecting the second hidden and output layer.

   /*
    * Activations for the input layer.
    */
   public static double[] inputActivations;

   /*
    * Activations for the first hidden layer.
    */
   public static double[] firstHiddenActivations;

   /*
    * Activations for the second hidden layer.
    */
   public static double[] secondHiddenActivations;

   /*
    * Activations for the output layer.
    */
   public static double[] outputActivations;    // Activations of output layer.

   public static int numCases;                  // Number of test cases.
   public static double[][] testCaseInputs;     // Inputs for each test case.
   public static double[][] testCaseOutputs;    // Outputs for each test case.

   public static double[] psiiValues;           // Values for each psii calculated when evaluating the network for training.
   public static double[] psijValues;           // Values for each psij calculated when evaluating the network for training.

   public static double[] thetakValues;         // Values for each theta k calculated when evaluating the network for training.
   public static double[] thetajValues;         // Values for each theta j calculated when evaluating the network for training.
   public static double[] thetaiValues;         // Values for each theta i calculated when evaluating the network for training.

   public static int inputNodes;                // Number of nodes in input layer.
   public static int firstHiddenNodes;          // Number of nodes in hidden layer.
   public static int secondHiddenNodes;         // Number of nodes in hidden layer.
   public static int outputNodes;               // Number of nodes in output layer.

   public static double lowerWeightBound;       // Lower bound for possible weight value.
   public static double upperWeightBound;       // Upper bound for possible weight value.

   public static double learningRate;           // Learning rate for training.

   public static int maxIters;                  // Maximum number of training iterations.
   public static double errorThreshold;         // Value of error for which training ends.

   public static int printStepSize;             // How often to print statistics during training.
   public static int weightStepSize;            // How often to save weights during training. -1 if only upon training completion.

   public static boolean preloadWeights;        // True if preloaded weights should be used during training, else false.
   public static boolean isTraining;            // True if training should be performed, false if running should be performed.

   public static String inputWeightsPath;       // Filepath to read in preloaded weights from.
   public static String outputWeightsPath;      // Filepath to output saved weights to.
   public static String testCasePath;           // Filepath to read in test cases from.

   public static long startingTime;             // Time at which running or training is started (in nanoseconds).

   /*
    * Can convert from nanoseconds to milliseconds by dividing by this constant.
    */
   public static final int NANO_TO_MILLI = 1000000;

   /*
    * Used to round number to 2 decimal places.
    */
   public static final int ROUNDING_VAL = 100;

   /*
    * Main method where the network is either run or trained.
    * @param args Command line inputs.
    */
   public static void main(String[] args)
   {
      config();

      if (isTraining)
         trainNetwork();
      else
         runNetwork();
   } // public static void main(String[] args)

   /*
    * Set configuration values for network.
    */
   public static void config()
   {
      startingTime = System.nanoTime();

      try
      {
         Scanner sc = new Scanner(new FileReader("files/config.txt"));

         sc.nextLine(); // Skip over header

         /*
          * Read in parameters of the network.
          */
         inputNodes = sc.nextInt();
         firstHiddenNodes = sc.nextInt();
         secondHiddenNodes = sc.nextInt();
         outputNodes = sc.nextInt();
         sc.nextLine(); // Skip over newline character.

         sc.nextLine();
         isTraining = Boolean.parseBoolean(sc.nextLine());

         sc.nextLine(); // Skip over line.
         lowerWeightBound = Double.parseDouble(sc.nextLine());

         sc.nextLine();
         upperWeightBound = Double.parseDouble(sc.nextLine());

         sc.nextLine();
         learningRate = Double.parseDouble(sc.nextLine());

         sc.nextLine();
         maxIters = Integer.parseInt(sc.nextLine());

         sc.nextLine();
         errorThreshold = Double.parseDouble(sc.nextLine());

         sc.nextLine();
         printStepSize = Integer.parseInt(sc.nextLine());

         sc.nextLine();
         weightStepSize = Integer.parseInt(sc.nextLine());

         sc.nextLine();
         preloadWeights = Boolean.parseBoolean(sc.nextLine());

         sc.nextLine();
         inputWeightsPath = sc.nextLine();

         sc.nextLine();
         outputWeightsPath = sc.nextLine();

         sc.nextLine();
         numCases = Integer.parseInt(sc.nextLine());

         sc.nextLine();
         testCasePath = sc.nextLine();
      } // try
      catch (Exception e)
      {
         e.printStackTrace();
      }
   } // public static void config()

   /*
    * Run the network.
    */
   public static void runNetwork()
   {
      System.out.println("RUNNING " + inputNodes + " BY " + firstHiddenNodes + " BY " +
              secondHiddenNodes + " BY " + outputNodes + " NETWORK.");
      System.out.println();

      allocateMemoryRun();
      loadValuesRun();

      /*
       * Calculate error of the network.
       */
      double error = 0.0;

      for (int testcase = 0; testcase < numCases; testcase++)
      {
         loadTestCase(testcase);
         evaluateNetworkRun();
         error += calculateError(testcase);
      } // for (int testcase = 0; testcase < numCases; testcase++)

      System.out.println("ERROR: " + error);
   } // public static void runNetwork()

   /*
    * Calculate outputs of network with given weights for the running process.
    */
   public static void evaluateNetworkRun()
   {
      /*
       * Calculate k layer activations by looping through each weight connecting the input layer the and first hidden layer.
       */
      for (int k = 0; k < firstHiddenNodes; k++)
      {
         double hiddenActivation = 0.0;
         for (int m = 0; m < inputNodes; m++)
         {
            double inputActivation = inputActivations[m];
            double weight = firstLayerWeights[m][k];

            hiddenActivation += inputActivation * weight;
         }
         firstHiddenActivations[k] = f(hiddenActivation);
      } // for (int j = 0; j < hiddenLayerNodes; j++)

      /*
       * Calculate j layer activations by looping through each weight connecting the first hidden layer the and next layer.
       */
      for (int j = 0; j < secondHiddenNodes; j++)
      {
         double secondHiddenActivation = 0.0;

         for (int k = 0; k < firstHiddenNodes; k++)
         {
            double firstHiddenActivation = firstHiddenActivations[k];
            double weight = secondLayerWeights[k][j];

            secondHiddenActivation += firstHiddenActivation * weight;
         } // for (int j = 0; j < hiddenLayerNodes; j++)

         secondHiddenActivations[j] = f(secondHiddenActivation);
      } // for (int i = 0; i < outputNodes; i++)

      /*
       * Calculate output layer activations by looping through each weight connecting the second hidden layer the and output layer.
       */
      for (int i = 0; i < outputNodes; i++)
      {
         double outputActivation = 0.0;

         for (int j = 0; j < secondHiddenNodes; j++)
         {
            double secondHiddenActivation = secondHiddenActivations[j];
            double weight = thirdLayerWeights[j][i];

            outputActivation += secondHiddenActivation * weight;
         } // for (int j = 0; j < hiddenLayerNodes; j++)

         outputActivations[i] = f(outputActivation);
      } // for (int i = 0; i < outputNodes; i++)
   } // public static void evaluateNetworkRun()

   /*
    * Allocate memory for weights, activations, and test cases arrays for the running process.
    */
   public static void allocateMemoryRun()
   {
      System.out.println("Allocating memory.");

      /*
       * Can be accessed in the form [m][k], where m is the index of the node in the input layer
       * and k is the index of the node in the first hidden layer.
       */
      firstLayerWeights = new double[inputNodes][firstHiddenNodes];

      /*
       * Can be accessed in the form [k][j], where k is the index of the node in the first hidden layer
       * and j is the index of the node in the second hidden layer.
       */
      secondLayerWeights = new double[firstHiddenNodes][secondHiddenNodes];

      /*
       * Can be accessed in the form [j][i], where j is the index of the node in the second hidden layer
       * and i is the index of the node in the output layer.
       */
      thirdLayerWeights = new double[secondHiddenNodes][outputNodes];

      inputActivations = new double[inputNodes];
      firstHiddenActivations = new double[firstHiddenNodes];
      secondHiddenActivations = new double[secondHiddenNodes];
      outputActivations = new double[outputNodes];

      testCaseInputs = new double[numCases][inputNodes];
      testCaseOutputs = new double[numCases][outputNodes];
   } // public static void allocateMemoryRun()

   /*
    * Allocate memory for weights, activations, and test cases arrays for the training process.
    */
   public static void allocateMemoryTrain()
   {
      System.out.println("Allocating memory.");

      /*
       * Can be accessed in the form [m][k], where m is the index of the node in the input layer
       * and k is the index of the node in the first hidden layer.
       */
      firstLayerWeights = new double[inputNodes][firstHiddenNodes];

      /*
       * Can be accessed in the form [k][j], where k is the index of the node in the first hidden layer
       * and j is the index of the node in the second hidden layer.
       */
      secondLayerWeights = new double[firstHiddenNodes][secondHiddenNodes];

      /*
       * Can be accessed in the form [j][i], where j is the index of the node in the second hidden layer
       * and i is the index of the node in the output layer.
       */
      thirdLayerWeights = new double[secondHiddenNodes][outputNodes];

      inputActivations = new double[inputNodes];
      firstHiddenActivations = new double[firstHiddenNodes];
      secondHiddenActivations = new double[secondHiddenNodes];
      outputActivations = new double[outputNodes];

      testCaseInputs = new double[numCases][inputNodes];
      testCaseOutputs = new double[numCases][outputNodes];

      psiiValues = new double[outputNodes];
      psijValues = new double[secondHiddenNodes];

      thetaiValues = new double[outputNodes];
      thetajValues = new double[secondHiddenNodes];
      thetakValues = new double[firstHiddenNodes];
   } // public static void allocateMemoryTrain()

   /*
    * Load in values for test input activations, expected outputs, and precalculated weights for the running process.
    */
   public static void loadValuesRun()
   {
      System.out.println("Loading values.");

      /*
       * Load in weights from a file.
       */
      loadWeights();

      /*
       * Load in the test cases from a file.
       */
      loadTruthTable();
   } // public static void loadValuesRun()

   /*
    * Load in values for test input activations and expected outputs during the training process.
    */
   public static void loadValuesTrain()
   {
      System.out.println("Loading values.");

      loadTruthTable();
   } // public static void loadValuesTrain()

   /*
    * Loads weights from a file into the network.
    */
   public static void loadWeights()
   {
      /*
       * Will attempt to read weights from the file into the network.
       *
       * If an Exception is caught, the exception will be printed.
       */
      try
      {
         Scanner sc = new Scanner(new File(inputWeightsPath));

         sc.nextLine(); // Skip over header

         /*
          * Verify that the parameters of the network are correct.
          */
         int inputNum = sc.nextInt();
         int firstHiddenNum = sc.nextInt();
         int secondHiddenNum = sc.nextInt();
         int outputNum = sc.nextInt();

         /*
          * Eat newline character.
          */
         sc.nextLine();

         /*
          * Printing an error message so the user can fix network configuration.
          */
         if (inputNum != inputNodes || firstHiddenNum != firstHiddenNodes || secondHiddenNum != secondHiddenNodes
                 || outputNum != outputNodes)
            System.out.println("\n" + "------------USER ERROR! The file configuration for the network's dimensions" +
                    "does not match the config.txt parameters.------------\n");

         /*
          * Skip over next four lines.
          */
         for (int line = 0; line < 4; line++)
            sc.nextLine();

         /*
          * Read in firstLayerWeights.
          */
         for (int line = 0; line < inputNodes * firstHiddenNodes; line++)
         {
            int m = sc.nextInt();
            int k = sc.nextInt();
            double weight = sc.nextDouble();

            /*
             * Eat newline character.
             */
            sc.nextLine();

            firstLayerWeights[m][k] = weight;
         } // for (int line = 0; line < inputNodes * firstHiddenNodes; line++)

         /*
          * Skip over next four lines.
          */
         for (int line = 0; line < 4; line++)
            sc.nextLine();

         /*
          * Read in secondLayerWeights.
          */
         for (int line = 0; line < firstHiddenNodes * secondHiddenNodes; line++)
         {
            int k = sc.nextInt();
            int j = sc.nextInt();
            double weight = sc.nextDouble();

            /*
             * Read newline character.
             */
            sc.nextLine();

            secondLayerWeights[k][j] = weight;
         } // for (int line = 0; line < firstHiddenNodes * secondHiddenNodes; line++)

         /*
          * Skip over next four lines.
          */
         for (int line = 0; line < 4; line++)
            sc.nextLine();

         /*
          * Read in thirdLayerWeights.
          */
         for (int line = 0; line < secondHiddenNodes * outputNodes; line++)
         {
            int j = sc.nextInt();
            int i = sc.nextInt();
            double weight = sc.nextDouble();

            /*
             * Read newline character.
             */
            sc.nextLine();

            thirdLayerWeights[j][i] = weight;
         } // for (int line = 0; line < secondHiddenNodes * outputNodes; line++)
      } // try
      catch (Exception e)
      {
         e.printStackTrace();
      }
   } // public static void loadWeights()

   /*
    * Saves the weights of the network to a file.
    * @param filename The name of the file to save to.
    */
   public static void saveWeights(String filename)
   {
      /*
       * Will attempt to save the weights.
       *
       * If an Exception is caught, the exception will be printed.
       */
      try
      {
         PrintWriter pw = new PrintWriter(new FileWriter(filename));

         /*
          * Print out parameters of the network.
          */
         pw.println("first_layer_nodes (second_layer_nodes third_layer_nodes ...)");
         pw.println(inputNodes + " " + firstHiddenNodes + " " + secondHiddenNodes + " " + outputNodes);
         pw.println();

         pw.println("first_layer_weights:");
         pw.println();
         pw.println("starting_node ending_node weight_value");

         for (int r = 0; r < inputNodes; r++)
            for (int c = 0; c < firstHiddenNodes; c++)
               pw.println(r + " " + c + " " + firstLayerWeights[r][c]);
         pw.println();

         pw.println("second_layer_weights:");
         pw.println();
         pw.println("starting_node ending_node weight_value");

         for (int r = 0; r < firstHiddenNodes; r++)
            for (int c = 0; c < secondHiddenNodes; c++)
               pw.println(r + " " + c + " " + secondLayerWeights[r][c]);
         pw.println();

         pw.println("third_layer_weights:");
         pw.println();
         pw.println("starting_node ending_node weight_value");

         for (int r = 0; r < secondHiddenNodes; r++)
            for (int c = 0; c < outputNodes; c++)
               pw.println(r + " " + c + " " + thirdLayerWeights[r][c]);

         pw.close();
      } // try
      catch (Exception e)
      {
         e.printStackTrace();
      }
   } // public static void saveWeights()

   /*
    * Randomizes the weights of the network.
    */
   public static void randomizeWeights()
   {
      System.out.println("Randomizing weights. ");

      /*
       * Randomize weights connecting m layer and k layer.
       */
      for (int m = 0; m < inputNodes; m++)
         for (int k = 0; k < firstHiddenNodes; k++)
            firstLayerWeights[m][k] = randomNumber(lowerWeightBound, upperWeightBound);

      /*
       * Randomize weights connecting k layer and j layer.
       */
      for (int k = 0; k < firstHiddenNodes; k++)
         for (int j = 0; j < secondHiddenNodes; j++)
            secondLayerWeights[k][j] = randomNumber(lowerWeightBound, upperWeightBound);

      /*
       * Randomize weights connecting j layer and i layer.
       */
      for (int j = 0; j < secondHiddenNodes; j++)
         for (int i = 0; i < outputNodes; i++)
            thirdLayerWeights[j][i] = randomNumber(lowerWeightBound, upperWeightBound);
   } // public static void randomizeWeights()

   /*
    * Load in the truth table that is used in running and training.
    */
   public static void loadTruthTable()
   {
      try
      {
         Scanner sc = new Scanner(new File(testCasePath));
         sc.nextLine();

         /*
          * Verify that number of input nodes and output nodes is correct.
          */
         int inpNodes = sc.nextInt();
         int outNodes = sc.nextInt();

         /*
          * Printing an error message if the number is incorrect, so the user can fix the network configuration.
          */
         if (inpNodes != inputNodes || outNodes != outputNodes)
            System.out.println("\n" + "------------USER ERROR! The file configuration for the network's dimensions" +
                    "does not match the config.txt parameters.------------\n");

         sc.nextLine();

         /*
          * Skip data header.
          */
         sc.nextLine();

         /*
          * Read in truth table.
          */
         for (int test = 0; test < numCases; test++)
         {
            /*
             * Read in inputs for testcase.
             */
            for (int inpVals = 0; inpVals < inputNodes; inpVals++)
            {
               double val = sc.nextDouble();
               testCaseInputs[test][inpVals] = val;
            }

            /*
             * Read in outputs for testcase.
             */
            for (int outVals = 0; outVals < outputNodes; outVals++)
            {
               double val = sc.nextDouble();
               testCaseOutputs[test][outVals] = val;
            }

            /*
             * Skip over newline character, except for last case.
             */
            if (test != numCases - 1)
               sc.nextLine();
         } // for (int test = 0; test < numCases; test++)
      } // try
      catch (Exception e)
      {
         e.printStackTrace();
      }
   } // public static void loadTruthTable()

   /*
    * Loads a test case, specified by an integer number, into the inputs of the network.
    * @param testCase The test case to be loaded into the network.
    */
   public static void loadTestCase(int testCase)
   {
      for (int inp = 0; inp < inputNodes; inp++)
         inputActivations[inp] = testCaseInputs[testCase][inp];
   }

   /*
    * Calculate the error in the network's output.
    * @param testCase Test case to calculate error upon.
    * @return the error in the network's output.
    */
   public static double calculateError(int testCase)
   {
      double error = 0.0;

      for (int i = 0; i < outputNodes; i++)
      {
         double omegai = testCaseOutputs[testCase][i] - outputActivations[i];
         error += omegai * omegai;
      }

      return error * 0.5;
   } // public static double calculateError(int testCase)

   /*
    * Prints a report summarizing the training process.
    * @param numIters Number of iterations used to train the network.
    * @param finalError The final error value the network reached.
    */
   public static void printReport(int numIters, double finalError)
   {
      System.out.println();
      System.out.println("================== TRAINING REPORT STARTING ==================");
      System.out.println();

      System.out.println(inputNodes + " by " + firstHiddenNodes + " by " +
              secondHiddenNodes + " by " + outputNodes + " network.");
      System.out.println();

      /*
       * Prints different messages depending on if the error threshold was reached.
       */
      if (numIters < maxIters)
         System.out.println("Error threshold reached. ");
      else
         System.out.println("Maximum number of iterations reached. ");

      System.out.println();
      System.out.println("Error: " + finalError);
      System.out.println("Iterations: " + numIters);
      System.out.println();

      /*
       * Print final outputs for test cases.
       */
      for (int testCase = 0; testCase < numCases; testCase++)
      {
         loadTestCase(testCase);
         evaluateNetworkRun();
         System.out.println("INPUT " + Arrays.toString(testCaseInputs[testCase])
               + " |  OUTPUT (F): " + Arrays.toString(outputActivations)
                 + ", Expected (T): " + Arrays.toString(testCaseOutputs[testCase]));
      } // for (int testCase = 0; testCase < numCases; testCase++)

      System.out.println();

      System.out.println("Random Number Range: [" + lowerWeightBound + ", " + upperWeightBound + ")");
      System.out.println("Max Iterations: " + maxIters);
      System.out.println("Error Threshold: " + errorThreshold);
      System.out.println("Learning Rate: " + learningRate);

      System.out.println();
      System.out.println("Weights Preloaded? " + preloadWeights);
      System.out.println("Weights Input File (if applicable): " + inputWeightsPath);
      System.out.println("Weights Output File: " + outputWeightsPath);
      System.out.println("Test Cases File: " + testCasePath);

      /*
       * Print time taken to complete training.
       */
      long endingTime = System.nanoTime();                                                      // Ending time (in nanoseconds).
      double completionTime = ((double) (endingTime - startingTime)) / NANO_TO_MILLI;           // Time for training (in millisec).
      double roundedTime = ((double) Math.round(completionTime * ROUNDING_VAL)) / ROUNDING_VAL; // Rounded time for training.

      System.out.println();
      System.out.println("Training Time: " + roundedTime + " milliseconds");

      System.out.println();
      System.out.println("================== TRAINING REPORT ENDING ==================");
   } // public static void printReport(int numIters, double finalError)

   /*
    * Trains the network by minimizing the error function that defines the difference between
    * the network's output for the given test cases and the expected output for those test cases.
    */
   public static void trainNetwork()
   {
      allocateMemoryTrain();
      loadValuesTrain();

      if (preloadWeights)
         loadWeights();
      else
         randomizeWeights();

      System.out.println("Training network. ");
      System.out.println();

      /*
       * Calculate initial error of network.
       */
      double error = 0.0;
      for (int testcase = 0; testcase < numCases; testcase++)
         error += calculateError(testcase);

      int iter = 0; // Training iteration counter.

      int weightIter = 0;

      /*
       * Train the network while the error threshold is not met or while the number of iterations
       * is less than the maximum number of possible training iterations.
       */
      while (error > errorThreshold && iter < maxIters)
      {
         error = 0.0; // Reset error for this training iteration.

         /*
          * Iterate through each test case and train network on each test case.
          */
         for (int testCase = 0; testCase < numCases; testCase++)
         {
            loadTestCase(testCase);
            evaluateNetworkTrain(testCase);

            error += calculateError(testCase);

            updateWeights();
            evaluateNetworkTrain(testCase);
         } // for (int testCase = 0; testCase < numCases; testCase++)

         iter++;
         weightIter++;

         if (iter % printStepSize == 0)
            System.out.println("Iteration #" + iter + " complete. Error: " + error);

         /*
          * Save weights periodically if user requested it.
          */
         if (weightStepSize != -1 && weightIter % weightStepSize == 0)
         {
            saveWeights(outputWeightsPath);
         }
      } // while (error > errorThreshold && iter < maxIters)

      printReport(iter, error);

      if (weightStepSize == -1)
         saveWeights(outputWeightsPath);
   } // public static void trainNetwork()

   /*
    * Calculate outputs of network with given weights for the training process. Unlike evaluateNetworkRun, this method stores
    * certain values when calculating activations in the network.
    *
    * @param testCase The test case for which the network is being evaluated on.
    */
   public static void evaluateNetworkTrain(int testCase)
   {
      /*
       * Calculate k layer activations by looping through each weight connecting the input layer the and first hidden layer.
       */
      for (int k = 0; k < firstHiddenNodes; k++)
      {
         double thetak = 0.0;

         for (int m = 0; m < inputNodes; m++)
         {
            double inputActivation = inputActivations[m];
            double weight = firstLayerWeights[m][k];

            thetak += inputActivation * weight;
         } // for (int m = 0; m < inputNodes; m++)
         firstHiddenActivations[k] = f(thetak);
         thetakValues[k] = thetak;
      } // for (int k = 0; k < firstHiddenNodes; k++)

      /*
       * Calculate j layer activations by looping through each weight connecting the first hidden layer the and next layer.
       */
      for (int j = 0; j < secondHiddenNodes; j++)
      {
         double thetaj = 0.0;

         for (int k = 0; k < firstHiddenNodes; k++)
         {
            double firstHiddenActivation = firstHiddenActivations[k];
            double weight = secondLayerWeights[k][j];

            thetaj += firstHiddenActivation * weight;
         } // for (int k = 0; k < firstHiddenNodes; k++)
         secondHiddenActivations[j] = f(thetaj);
         thetajValues[j] = thetaj;
      } // for (int j = 0; j < secondHiddenNodes; j++)

      /*
       * Calculate output layer activations by looping through each weight connecting the second hidden layer the and output layer.
       */
      for (int i = 0; i < outputNodes; i++)
      {
         double thetai = 0.0;

         for (int j = 0; j < secondHiddenNodes; j++)
         {
            double secondHiddenActivation = secondHiddenActivations[j];
            double weight = thirdLayerWeights[j][i];

            thetai += secondHiddenActivation * weight;
         } // for (int j = 0; j < secondHiddenNodes; j++)

         thetaiValues[i] = thetai;

         double ai = f(thetai);
         outputActivations[i] = ai;

         double Ti = testCaseOutputs[testCase][i];
         double psii = (Ti - ai) * fPrime(thetai);
         psiiValues[i] = psii;
      } // for (int i = 0; i < outputNodes; i++)
   } // public static void evaluateNetworkTrain(int testCase)

   /*
    * Updates the weights of the network to minimize the error of the network's output layer with backpropagation.
    */
   public static void updateWeights()
   {
      for (int j = 0; j < secondHiddenNodes; j++)
      {
         double omegaj = 0.0;

         for (int i = 0; i < outputNodes; i++)
         {
            double psii = psiiValues[i];
            omegaj += psii * thirdLayerWeights[j][i];

            double weightChangeji = learningRate * secondHiddenActivations[j] * psii;

            thirdLayerWeights[j][i] += weightChangeji;    // Store changes for each weight.
         } // for (int i = 0; i < outputNodes; i++)

         double thetaj = thetajValues[j];
         double psij = omegaj * fPrime(thetaj);
         psijValues[j] = psij;
      } // for (int j = 0; j < secondHiddenNodes; j++)

      for (int k = 0; k < firstHiddenNodes; k++)
      {
         double omegak = 0.0;

         for (int j = 0; j < secondHiddenNodes; j++)
         {
            double psij = psijValues[j];
            omegak += psij * secondLayerWeights[k][j];

            double weightChangekj = learningRate * firstHiddenActivations[k] * psij;

            secondLayerWeights[k][j] += weightChangekj;    // Store changes for each weight.
         } // for (int j = 0; j < secondHiddenNodes; j++)

         double thetak = thetakValues[k];
         double psik = omegak * fPrime(thetak);

         for (int m = 0; m < inputNodes; m++)
         {
            firstLayerWeights[m][k] += learningRate * inputActivations[m] * psik;
         }
      } // for (int k = 0; k < firstHiddenNodes; k++)
   } // public static void updateWeights()

   /*
    * Generate a random number between certain bounds: [lower, upper).
    * @param low Lower bound.
    * @param high Upper bound.
    * @return a random number between the specified bounds.
    */
   public static double randomNumber(double low, double high)
   {
      return (high - low) * Math.random() + low;
   } // public static double randomNumber(double low, double high)

   /*
    * The sigmoid activation function.
    * @param num number to input.
    * @return output of activation function.
    */
   public static double f(double num)
   {
      return 1.0 / (1.0 + Math.exp(-num));
   } // public static double f(double num)

   /*
    * Derivative of the sigmoid activation function.
    * @param num number to input.
    * @return output of derivative of activation function.
    */
   public static double fPrime(double num)
   {
      double output = f(num);
      return output * (1.0 - output);
   } // public static double fPrime(double num)
} // public class Network