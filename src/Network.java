import java.io.*;
import java.util.Arrays;
import java.util.Scanner;

/*
 * This class represents an A by B by C perceptron. There are A neurons in the input layer,
 * B neurons in the hidden layer, and C neurons in the output layer.
 *
 * The network takes in sets of input test cases and can train itself with backpropagation to predict the outputs of
 * those test cases. Floating point weights connect the neurons between each layer of the network.
 *
 * @author Anirudh Kotamraju
 * @version March 29, 2022
 */
public class Network
{
   /*
    * Static variables.
    */

   public static double[][] firstLayerWeights;  // Weights connecting the input and hidden layer.
   public static double[][] secondLayerWeights; // Weights connecting the hidden and output layer.

   /*
    * Changes in weights between each training iteration.
    */
   public static double[][] firstLayerWeightChanges;

   /*
    * Changes in weights between each training iteration.
    */
   public static double[][] secondLayerWeightChanges;

   public static double[] inputActivations;     // Activations of input layer.
   public static double[] hiddenActivations;    // Activations of hidden layer.
   public static double[] outputActivations;    // Activations of output layer.

   public static int numCases;                  // Number of test cases.
   public static double[][] testCaseInputs;     // Inputs for each test case.
   public static double[][] testCaseOutputs;    // Outputs for each test case.

   public static double[] psiiValues;           // Values for each psii calculated when evaluating the network for training.
   public static double[] omegajValues;         // Values for each omegaj calculated when evaluating the network for training.
   public static double[] thetajValues;         // Values for each theta j calculated when evaluating the network for training.
   public static double[] thetaiValues;         // Values for each theta i calculated when evaluating the network for training.

   public static int inputNodes;                // Number of nodes in input layer.
   public static int hiddenLayerNodes;          // Number of nodes in hidden layer.
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
    * Can convert from nanoseconds to seconds by dividing by this constant.
    */
   public static final int NANO_TO_SECONDS = 1000000000;

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
         hiddenLayerNodes = sc.nextInt();
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
      System.out.println("RUNNING " + inputNodes + " BY " + hiddenLayerNodes + " BY " + outputNodes + " NETWORK.");
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
       * Calculate hidden layer activations by looping through each weight connecting the input layer the and hidden layer.
       */
      for (int j = 0; j < hiddenLayerNodes; j++)
      {
         double hiddenActivation = 0.0;
         for (int k = 0; k < inputNodes; k++)
         {
            double inputActivation = inputActivations[k];
            double weight = firstLayerWeights[k][j];

            hiddenActivation += inputActivation * weight;
         }
         hiddenActivations[j] = f(hiddenActivation);
      } // for (int j = 0; j < hiddenLayerNodes; j++)

      /*
       * Calculate output layer activations by looping through each weight connecting the hidden layer the and output layer.
       */
      for (int i = 0; i < outputNodes; i++)
      {
         double outputActivation = 0.0;

         for (int j = 0; j < hiddenLayerNodes; j++)
         {
            double hiddenActivation = hiddenActivations[j];
            double weight = secondLayerWeights[j][i];

            outputActivation += hiddenActivation * weight;
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
       * Can be accessed in the form [k][j], where k is the index of the node in the input layer
       * and j is the index of the node in the hidden layer.
       */
      firstLayerWeights = new double[inputNodes][hiddenLayerNodes];

      /*
       * Can be accessed in the form [j][i], where j is the index of the node in the hidden layer
       * and i is the index of the node in the output layer.
       */
      secondLayerWeights = new double[hiddenLayerNodes][outputNodes];

      inputActivations = new double[inputNodes];
      hiddenActivations = new double[hiddenLayerNodes];
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
       * Can be accessed in the form [k][j], where k is the index of the node in the input layer
       * and j is the index of the node in the hidden layer.
       */
      firstLayerWeights = new double[inputNodes][hiddenLayerNodes];

      /*
       * Can be accessed in the form [j][i], where j is the index of the node in the hidden layer
       * and i is the index of the node in the output layer.
       */
      secondLayerWeights = new double[hiddenLayerNodes][outputNodes];

      firstLayerWeightChanges = new double[inputNodes][hiddenLayerNodes];
      secondLayerWeightChanges = new double[hiddenLayerNodes][outputNodes];

      inputActivations = new double[inputNodes];
      hiddenActivations = new double[hiddenLayerNodes];
      outputActivations = new double[outputNodes];

      testCaseInputs = new double[numCases][inputNodes];
      testCaseOutputs = new double[numCases][outputNodes];

      psiiValues = new double[outputNodes];
      omegajValues = new double[hiddenLayerNodes];
      thetaiValues = new double[outputNodes];
      thetajValues = new double[hiddenLayerNodes];
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
         int hiddenNum = sc.nextInt();
         int outputNum = sc.nextInt();

         /*
          * Eat newline character.
          */
         sc.nextLine();

         /*
          * Printing an error message so the user can fix network configuration.
          */
         if (inputNum != inputNodes || hiddenNum != hiddenLayerNodes || outputNum != outputNodes)
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
         for (int line = 0; line < inputNodes * hiddenLayerNodes; line++)
         {
            int k = sc.nextInt();
            int j = sc.nextInt();
            double weight = sc.nextDouble();

            /*
             * Eat newline character.
             */
            sc.nextLine();

            firstLayerWeights[k][j] = weight;
         } // for (int line = 0; line < inputNodes * hiddenLayerNodes; line++)

         /*
          * Skip over next four lines.
          */
         for (int line = 0; line < 4; line++)
            sc.nextLine();

         /*
          * Read in secondLayerWeights.
          */
         for (int line = 0; line < hiddenLayerNodes * outputNodes; line++)
         {
            int j = sc.nextInt();
            int i = sc.nextInt();
            double weight = sc.nextDouble();

            /*
             * Read newline character.
             */
            sc.nextLine();

            secondLayerWeights[j][i] = weight;
         } // for (int line = 0; line < hiddenLayerNodes * outputNodes; line++)
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
         pw.println(inputNodes + " " + hiddenLayerNodes + " " + outputNodes);
         pw.println();

         pw.println("first_layer_weights:");
         pw.println();
         pw.println("starting_node ending_node weight_value");

         for (int r = 0; r < firstLayerWeights.length; r++)
            for (int c = 0; c < firstLayerWeights[0].length; c++)
               pw.println(r + " " + c + " " + firstLayerWeights[r][c]);
         pw.println();

         pw.println("second_layer_weights:");
         pw.println();
         pw.println("starting_node ending_node weight_value");

         for (int r = 0; r < secondLayerWeights.length; r++)
            for (int c = 0; c < secondLayerWeights[0].length; c++)
               pw.println(r + " " + c + " " + secondLayerWeights[r][c]);

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
       * Randomize weights connecting input layer and hidden layer.
       */
      for (int k = 0; k < inputNodes; k++)
         for (int j = 0; j < hiddenLayerNodes; j++)
            firstLayerWeights[k][j] = randomNumber(lowerWeightBound, upperWeightBound);

      /*
       * Randomize weights connecting hidden layer and output layer.
       */
      for (int j = 0; j < hiddenLayerNodes; j++)
         for (int i = 0; i < outputNodes; i++)
            secondLayerWeights[j][i] = randomNumber(lowerWeightBound, upperWeightBound);
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
          * Verify that input nodes and output nodes matches.
          */
         int inpNodes = sc.nextInt();
         int outNodes = sc.nextInt();

         /*
          * Printing an error message so the user can fix network configuration.
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

      System.out.println(inputNodes + " by " + hiddenLayerNodes + " by " + outputNodes + " network.");
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
      long endingTime = System.nanoTime();                                                 // Ending time (in nanoseconds).
      double completionTime = ((double) (endingTime - startingTime)) / NANO_TO_SECONDS;    // Time for training (in seconds).
      double roundedTime = ((double) Math.round(completionTime * 1000)) / 100;             // Rounded time for training.

      System.out.println();
      System.out.println("Training Time: " + roundedTime + " seconds");

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
       * Calculate hidden layer activations by looping through each weight connecting the input layer the and hidden layer.
       */
      for (int j = 0; j < hiddenLayerNodes; j++)
      {
         double thetaj = 0.0;

         for (int k = 0; k < inputNodes; k++)
         {
            double inputActivation = inputActivations[k];
            double weight = firstLayerWeights[k][j];

            thetaj += inputActivation * weight;
         }

         thetajValues[j] = thetaj;
         hiddenActivations[j] = f(thetaj);
      } // for (int j = 0; j < hiddenLayerNodes; j++)

      /*
       * Calculate output layer activations by looping through each weight connecting the hidden layer the and output layer.
       */
      for (int i = 0; i < outputNodes; i++)
      {
         double thetai = 0.0;

         for (int j = 0; j < hiddenLayerNodes; j++)
         {
            double hiddenActivation = hiddenActivations[j];
            double weight = secondLayerWeights[j][i];

            thetai += hiddenActivation * weight;
         } // for (int j = 0; j < hiddenLayerNodes; j++)

         thetaiValues[i] = thetai;

         double Fi = f(thetai);
         outputActivations[i] = Fi;

         double omegai = testCaseOutputs[testCase][i] - Fi;
         double psii = omegai * fPrime(thetai);
         psiiValues[i] = psii;
      } // for (int i = 0; i < outputNodes; i++)
   } // public static void evaluateNetworkTrain()

   /*
    * Updates the weights of the network to minimize the error of the network's output layer for
    * a given test case.
    */
   public static void updateWeights()
   {
      /*
       * Updating weight connecting hidden node j and output node i.
       */
      for (int j = 0; j < hiddenLayerNodes; j++)
      {
         double omegaj = 0.0;

         for (int i = 0; i < outputNodes; i++)
         {
            double psii = psiiValues[i];
            omegaj += psii * secondLayerWeights[j][i];

            double weightChangeji = learningRate * hiddenActivations[j] * psii;

            secondLayerWeights[j][i] += weightChangeji;    // Store changes for each weight.
         } // for (int i = 0; i < outputNodes; i++)

         omegajValues[j] = omegaj;                         // Store omegaj values, which are needed for updating kj weights.
      } // for (int j = 0; j < hiddenLayerNodes; j++)

      /*
       * Updating weight connecting input node k and hidden node j.
       */
      for (int k = 0; k < inputNodes; k++)
      {
         for (int j = 0; j < hiddenLayerNodes; j++)
         {
            double thetaj = thetajValues[j];
            double upperPsij = omegajValues[j] * fPrime(thetaj);

            double weightChangekj = learningRate * inputActivations[k] * upperPsij;

            firstLayerWeights[k][j] += weightChangekj; // Store changes for each weight.
         } // for (int j = 0; j < hiddenLayerNodes; j++)
      } // for (int k = 0; k < inputNodes; k++)
   } // public static void updateWeights(int testCase)

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