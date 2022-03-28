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
 * TODO:
 *   - add time to complete train/run
 *   - checking documentation for everything
 *   - ask if it's okay if we instantiate BR/PW in load/saveWeights methods
 *   - ask if it's okay if we just put filename in allocate there
 *   - check about conditional for preloading vs randomizing weights
 *   - ask about config file structure
 *   - running vs. training in config file too?
 *
 * @author Anirudh Kotamraju
 * @version March 28, 2022
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

   public static boolean preloadWeights;        // True if preloaded weights should be used during training, else false.
   public static String inputWeightsPath;       // Filepath to read in preloaded weights from.
   public static String outputWeightsPath;      // Filepath to output saved weights to.

   /*
    * Main method where the network is either run or trained.
    * @param args Command line inputs.
    */
   public static void main(String[] args)
   {
      boolean run = false;

      config();

      if (run)
         runNetwork();
      else
         trainNetwork();
   } // public static void main(String[] args)

   /*
    * Set configuration values for network.
    */
   public static void config()
   {
      numCases = 4;

      try
      {
         BufferedReader br = new BufferedReader(new FileReader("files/config.txt"));

         br.readLine(); // Skip over header

         /*
          * Read in parameters of the network.
          */
         String str = br.readLine();
         String[] params = str.split(" ");
         inputNodes = Integer.parseInt(params[0]);
         hiddenLayerNodes = Integer.parseInt(params[1]);
         outputNodes = Integer.parseInt(params[2]);

         br.readLine(); // Skip over line.
         lowerWeightBound = Double.parseDouble(br.readLine());

         br.readLine(); // Skip over line.
         upperWeightBound = Double.parseDouble(br.readLine());

         br.readLine(); // Skip over line.
         learningRate = Double.parseDouble(br.readLine());

         br.readLine(); // Skip over line.
         maxIters = Integer.parseInt(br.readLine());

         br.readLine(); // Skip over line.
         errorThreshold = Double.parseDouble(br.readLine());

         br.readLine(); // Skip over line.
         printStepSize = Integer.parseInt(br.readLine());

         br.readLine();
         preloadWeights = Boolean.parseBoolean(br.readLine());

         br.readLine();
         inputWeightsPath = br.readLine();

         br.readLine();
         outputWeightsPath = br.readLine();
      } // try
      catch (Exception e)
      {
         e.printStackTrace();
      }
   } // public static void config()

   /*
    * Run the network with preloaded weights.
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
       * First test case inputs.
       */
      testCaseInputs[0][0] = 0.0;
      testCaseInputs[0][1] = 0.0;

      /*
       * First test case outputs.
       */
      testCaseOutputs[0][0] = 0.0;              // OR
      testCaseOutputs[0][1] = 0.0;              // AND
      testCaseOutputs[0][2] = 0.0;              // XOR

      /*
       * Second test case inputs.
       */
      testCaseInputs[1][0] = 0.0;
      testCaseInputs[1][1] = 1.0;

      /*
       * Second test case outputs.
       */
      testCaseOutputs[1][0] = 1.0;              // OR
      testCaseOutputs[1][1] = 0.0;              // AND
      testCaseOutputs[1][2] = 1.0;              // XOR

      /*
       * Third test case inputs.
       */
      testCaseInputs[2][0] = 1.0;
      testCaseInputs[2][1] = 0.0;

      /*
       * Third test case outputs.
       */
      testCaseOutputs[2][0] = 1.0;              // OR
      testCaseOutputs[2][1] = 0.0;              // AND
      testCaseOutputs[2][2] = 1.0;              // XOR

      /*
       * Fourth test case inputs.
       */
      testCaseInputs[3][0] = 1.0;
      testCaseInputs[3][1] = 1.0;

      /*
       * Fourth test case outputs.
       */
      testCaseOutputs[3][0] = 1.0;              // OR
      testCaseOutputs[3][1] = 1.0;              // AND
      testCaseOutputs[3][2] = 0.0;              // XOR
   } // public static void loadValuesRun()

   /*
    * Load in values for test input activations and expected outputs during the training process.
    */
   public static void loadValuesTrain()
   {
      System.out.println("Loading values.");

      /*
       * First test case inputs.
       */
      testCaseInputs[0][0] = 0.0;
      testCaseInputs[0][1] = 0.0;

      /*
       * First test case outputs.
       */
      testCaseOutputs[0][0] = 0.0;              // OR
      testCaseOutputs[0][1] = 0.0;              // AND
      testCaseOutputs[0][2] = 0.0;              // XOR

      /*
       * Second test case inputs.
       */
      testCaseInputs[1][0] = 0.0;
      testCaseInputs[1][1] = 1.0;

      /*
       * Second test case outputs.
       */
      testCaseOutputs[1][0] = 1.0;              // OR
      testCaseOutputs[1][1] = 0.0;              // AND
      testCaseOutputs[1][2] = 1.0;              // XOR

      /*
       * Third test case inputs.
       */
      testCaseInputs[2][0] = 1.0;
      testCaseInputs[2][1] = 0.0;

      /*
       * Third test case outputs.
       */
      testCaseOutputs[2][0] = 1.0;              // OR
      testCaseOutputs[2][1] = 0.0;              // AND
      testCaseOutputs[2][2] = 1.0;              // XOR

      /*
       * Fourth test case inputs.
       */
      testCaseInputs[3][0] = 1.0;
      testCaseInputs[3][1] = 1.0;

      /*
       * Fourth test case outputs.
       */
      testCaseOutputs[3][0] = 1.0;              // OR
      testCaseOutputs[3][1] = 1.0;              // AND
      testCaseOutputs[3][2] = 0.0;              // XOR
   } // public static void loadValuesTrain()

   /*
    * Calculate the error in the network's output.
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
      System.out.println("Weights Input FilePath (if applicable): " + inputWeightsPath);
      System.out.println("Weights Output FilePath: " + outputWeightsPath);

      System.out.println();
      System.out.println("================== TRAINING REPORT ENDING ==================");
   } // public static void printReport(int numIters, double finalError)

   /*
    * Loads weights from a file into the network.
    * @param filename The name of the file to read.
    */
   public static void loadWeights()
   {
      /*
       * Will attempt to read weights from the file into the network.
       *
       * If an IO Exception is caught, the exception will be printed.
       */
      try
      {
         BufferedReader br = new BufferedReader(new FileReader(inputWeightsPath));

         br.readLine(); // Skip over header

         /*
          * Verify that the parameters of the network are correct.
          */
         String str = br.readLine();
         String[] params = str.split(" ");
         int inputNum = Integer.parseInt(params[0]);
         int hiddenNum = Integer.parseInt(params[1]);
         int outputNum = Integer.parseInt(params[2]);

         /*
          * Printing an error message so the user can make
          */
         if (inputNum != inputNodes || hiddenNum != hiddenLayerNodes || outputNum != outputNodes)
            System.out.println("\n" + "------------USER ERROR! The file configuration for the network's dimensions" +
                    "does not match the config() parameters.------------\n");

         /*
          * Skip over next four lines.
          */
         for (int line = 0; line < 4; line++)
            br.readLine();

         /*
          * Read in firstLayerWeights.
          */
         for (int line = 0; line < inputNodes * hiddenLayerNodes; line++)
         {
            String[] vals = br.readLine().split(" ");
            int k = Integer.parseInt(vals[0]);
            int j = Integer.parseInt(vals[1]);
            double weight = Double.parseDouble(vals[2]);

            firstLayerWeights[k][j] = weight;
         } // for (int line = 0; line < inputNodes * hiddenLayerNodes; line++)

         /*
          * Skip over next four lines.
          */
         for (int line = 0; line < 4; line++)
            br.readLine();

         /*
          * Read in secondLayerWeights.
          */
         for (int line = 0; line < hiddenLayerNodes * outputNodes; line++)
         {
            String[] vals = br.readLine().split(" ");
            int j = Integer.parseInt(vals[0]);
            int i = Integer.parseInt(vals[1]);
            double weight = Double.parseDouble(vals[2]);

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
   public static void saveWeights()
   {
      /*
       * Will attempt to save the weights.
       *
       * If an Exception is caught, the exception will be printed.
       */
      try
      {
         PrintWriter pw = new PrintWriter(new FileWriter(outputWeightsPath));

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
    * Loads a test case, specified by an integer number, into the inputs of the network.
    * @param testCase The test case to be loaded into the network.
    */
   public static void loadTestCase(int testCase)
   {
      for (int inp = 0; inp < inputNodes; inp++)
         inputActivations[inp] = testCaseInputs[testCase][inp];
   }

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
       * Calculate initial error of network with randomized weights.
       */
      double error = 0.0;
      for (int testcase = 0; testcase < numCases; testcase++)
         error += calculateError(testcase);

      int iter = 0; // Training iteration counter.

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
            updateWeights();
            evaluateNetworkTrain(testCase);

            error += calculateError(testCase);
         } // for (int testCase = 0; testCase < numCases; testCase++)

         iter++;

         if (iter % printStepSize == 0)
         {
            System.out.println("Iteration #" + iter + " complete. Error: " + error);
         }

      } // while (error > errorThreshold && iter < maxIters)

      printReport(iter, error);

      saveWeights();
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