import java.util.Arrays;

/**
 * This class represents an A by B by 1 perceptron. There are A neurons in the input layer,
 * B neurons in the hidden layer, and 1 neuron in the output layer.
 *
 * The network takes in sets of input test cases and can train itself to predict the outputs of
 * those test cases. Floating point weights connect the neurons between each layer of the network.
 *
 *
 * @author Anirudh Kotamraju
 * @version February 28, 2022.
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

   public static int inputNodes;                // Number of nodes in input layer.
   public static int hiddenLayerNodes;          // Number of nodes in hidden layer.
   public static int outputNodes;               // Number of nodes in output layer.


   public static double lowerWeightBound;       // Lower bound for possible weight value.
   public static double upperWeightBound;       // Upper bound for possible weight value.

   public static double learningRate;           // Learning rate for training.

   public static int maxIters;                  // Maximum number of training iterations.
   public static double errorThreshold;         // Value of error for which training ends.

   /*
    * These are weights previously found to minimize error for the XOR problem for a 2 x 2 x 1 network.
    *
    * These are used in the "run" process for the network and will be removed once weights for the process
    * are read from a file in a future iteration of this codebase.
    */
   public static final double W000 = 0.947;
   public static final double W001 = 8.008;
   public static final double W010 = 0.947;
   public static final double W011 = 7.999;

   public static final double W100 = -37.284;
   public static final double W110 = 29.768;

   /*
    * How often to print statistics during training.
    */
   public static final int PRINT_STEP_SIZE = 1000;

   /**
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

   /**
    * Set configuration values for network.
    */
   public static void config()
   {
      inputNodes = 2;
      hiddenLayerNodes = 2;
      outputNodes = 1;

      numCases = 4;

      lowerWeightBound = 0.1;
      upperWeightBound = 1.5;

      learningRate = 0.3;
      maxIters = 100000;
      errorThreshold = 0.005;
   } // public static void config()

   /**
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
         evaluateNetwork();
         error += calculateError(testcase);
      } // for (int testcase = 0; testcase < numCases; testcase++)

      System.out.println("ERROR: " + error);
   } // public static void runNetwork()

   /**
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

   /**
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
       * Can be accessed in the form [j][i], where j is the index of the node in the input layer
       * and j is the index of the node in the hidden layer.
       */
      secondLayerWeights = new double[hiddenLayerNodes][outputNodes];

      firstLayerWeightChanges = new double[inputNodes][hiddenLayerNodes];
      secondLayerWeightChanges = new double[hiddenLayerNodes][outputNodes];

      inputActivations = new double[inputNodes];
      hiddenActivations = new double[hiddenLayerNodes];
      outputActivations = new double[outputNodes];

      testCaseInputs = new double[numCases][inputNodes];
      testCaseOutputs = new double[numCases][outputNodes];
   } // public static void allocateMemoryTrain()

   /**
    * Load in values for test input activations, expected outputs, and precalculated weights for the running process.
    */
   public static void loadValuesRun()
   {
      System.out.println("Loading values.");

      /*
       * Preload 2 by 2 by 1 network with weights previously found to minimize error for the XOR problem.
       * In the future, weights will be read from a file.
       */
      firstLayerWeights[0][0] = W000;
      firstLayerWeights[0][1] = W001;
      firstLayerWeights[1][0] = W010;
      firstLayerWeights[1][1] = W011;

      secondLayerWeights[0][0] = W100;
      secondLayerWeights[1][0] = W110;

      /*
       * First test case.
       */
      testCaseInputs[0][0] = 0.0;
      testCaseInputs[0][1] = 0.0;
      testCaseOutputs[0][0] = 0.0;

      /*
       * Second test case.
       */
      testCaseInputs[1][0] = 0.0;
      testCaseInputs[1][1] = 1.0;
      testCaseOutputs[1][0] = 1.0;

      /*
       * Third test case.
       */
      testCaseInputs[2][0] = 1.0;
      testCaseInputs[2][1] = 0.0;
      testCaseOutputs[2][0] = 1.0;

      /*
       * Fourth test case.
       */
      testCaseInputs[3][0] = 1.0;
      testCaseInputs[3][1] = 1.0;
      testCaseOutputs[3][0] = 1.0;
   } // public static void loadValuesRun()

   /**
    * Load in values for test input activations and expected outputs during the training process.
    */
   public static void loadValuesTrain()
   {
      System.out.println("Loading values.");

      /*
       * First test case.
       */
      testCaseInputs[0][0] = 0.0;
      testCaseInputs[0][1] = 0.0;
      testCaseOutputs[0][0] = 0.0;

      /*
       * Second test case.
       */
      testCaseInputs[1][0] = 0.0;
      testCaseInputs[1][1] = 1.0;
      testCaseOutputs[1][0] = 1.0;

      /*
       * Third test case.
       */
      testCaseInputs[2][0] = 1.0;
      testCaseInputs[2][1] = 0.0;
      testCaseOutputs[2][0] = 1.0;

      /*
       * Fourth test case.
       */
      testCaseInputs[3][0] = 1.0;
      testCaseInputs[3][1] = 1.0;
      testCaseOutputs[3][0] = 1.0;
   } // public static void loadValuesTrain()

   /**
    * Calculate output of network with given weights.
    */
   public static void evaluateNetwork()
   {
      /*
       * Update hidden layer activations by looping through each weight connecting the input layer the and hidden layer.
       */
      for (int j = 0; j < hiddenLayerNodes; j++)
      {
         /*
          * Calculate the new activation for the jth hidden layer node.
          */
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
       * Calculate new activation for the output layer node.
       */
      double outputActivation = 0.0;
      for (int j = 0; j < hiddenLayerNodes; j++)
      {
         double hiddenActivation = hiddenActivations[j];
         double weight = secondLayerWeights[j][0];

         outputActivation += hiddenActivation * weight;
      }
      outputActivations[0] = f(outputActivation);
   } // public static void evaluateNetwork()

   /**
    * Calculate the error in the network's output.
    * @return the error in the network's output.
    */
   public static double calculateError(int testCase)
   {
      return 0.5 * (testCaseOutputs[testCase][0] - outputActivations[0])
              * (testCaseOutputs[testCase][0] - outputActivations[0]);
   } // public static double calculateError(int testCase)


   /**
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
         evaluateNetwork();
         System.out.println("INPUT " + Arrays.toString(testCaseInputs[testCase])
               + " |  OUTPUT (F): " + outputActivations[0]
                 + ", Expected (T): " + testCaseOutputs[testCase][0]);
      } // for (int testCase = 0; testCase < numCases; testCase++)
      System.out.println();

      System.out.println("Random Number Range: [" + lowerWeightBound + ", " + upperWeightBound + ")");
      System.out.println("Max Iterations: " + maxIters);
      System.out.println("Error Threshold: " + errorThreshold);
      System.out.println("Learning Rate: " + learningRate);

      System.out.println();
      System.out.println("================== TRAINING REPORT ENDING ==================");
   } // public static void printReport(int numIters, double finalError)


   /**
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
       * Randomize weights connecting hidden layer and output layers
       */
      for (int j = 0; j < hiddenLayerNodes; j++)
         secondLayerWeights[j][0] = randomNumber(lowerWeightBound, upperWeightBound);
   } // public static void randomizeWeights()

   /**
    * Loads a test case, specified by an integer number, into the inputs of the network.
    * @param testCase The test case to be loaded into the network.
    */
   public static void loadTestCase(int testCase)
   {
      for (int inp = 0; inp < inputNodes; inp++)
         inputActivations[inp] = testCaseInputs[testCase][inp];
   }

   /**
    * Trains the network by minimizing the error function that defines the difference between
    * the network's output for the given test cases and the expected output for those test cases.
    */
   public static void trainNetwork()
   {
      allocateMemoryTrain();
      loadValuesTrain();
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
            evaluateNetwork();
            updateWeights(testCase);
            evaluateNetwork();

            error += calculateError(testCase);
         }

         iter++;

         if (iter % PRINT_STEP_SIZE == 0)
         {
            System.out.println("Iteration #" + iter + " complete. Error: " + error);
         }

      } // while (error > errorThreshold && iter < maxIters)
      printReport(iter, error);
   } // public static void trainNetwork()

   /**
    * Updates the weights of the network to minimize the error of the network's output layer for
    * a given test case.
    * @param testCase Test case for which the weights of the network should be updated.
    */
   public static void updateWeights(int testCase)
   {
      /*
       * Calculate theta0 by iterating through weights connecting hidden layer and output layer.
       */
      double theta0 = 0.0;
      for (int j = 0; j < hiddenLayerNodes; j++)
         theta0 += hiddenActivations[j] * secondLayerWeights[j][0];

      double F0 = f(theta0);                            // Network output for current test case.
      double T0 = testCaseOutputs[testCase][0];         // Expected output of test case.

      double omega0 = (T0 - F0);                        // Diff between network and expected outputs.
      double psi0 = omega0 * fPrime(theta0);

      /*
       * Calculate the new values for the weights connecting the hidden and output layers.
       */
      for (int j = 0; j < hiddenLayerNodes; j++)
      {
         double partial = -hiddenActivations[j] * psi0;

         double weightChange = -learningRate * partial;

         secondLayerWeightChanges[j][0] = weightChange; // Store changes for each weight.
      }

      /*
       * Calculate the new values for the weights connecting the input and hidden layers.
       */
      for (int j = 0; j < hiddenLayerNodes; j++)
      {
         /*
          * Calculate thetaj for the jth hidden layer node.
          */
         double thetaj = 0.0;
         for (int k = 0; k < inputNodes; k++)
            thetaj += inputActivations[k] * firstLayerWeights[k][j];

         /*
          * Storing change for weight connecting input node k and hidden node j.
          */
         for (int k = 0; k < inputNodes; k++)
         {
            double upperOmegaj = psi0 * secondLayerWeights[j][0];

            double upperPsij = upperOmegaj * fPrime(thetaj);

            double partial = -inputActivations[k] * upperPsij;
            double weightChange = -learningRate * partial;

            firstLayerWeightChanges[k][j] = weightChange; // Store changes for each weight.
         } // for (int k = 0; k < inputNodes; k++)
      } // for (int j = 0; j < hiddenLayerNodes; j++)


      /*
       * Update weights connecting hidden layer and output layers.
       */
      for (int j = 0; j < hiddenLayerNodes; j++)
         secondLayerWeights[j][0] += secondLayerWeightChanges[j][0];

      /*
       * Update weights connecting input layer and hidden layers.
       */
      for (int k = 0; k < inputNodes; k++)
         for (int j = 0; j < hiddenLayerNodes; j++)
            firstLayerWeights[k][j] += firstLayerWeightChanges[k][j];
   } // public static void updateWeights(int testCase)


   /**
    * Generate a random number between certain bounds: [lower, upper).
    * @param low Lower bound.
    * @param high Upper bound.
    * @return a random number between the specified bounds.
    */
   public static double randomNumber(double low, double high)
   {
      return (high - low) * Math.random() + low;
   } // public static double randomNumber(double low, double high)

   /**
    * The sigmoid activation function.
    * @param num number to input.
    * @return output of activation function.
    */
   public static double f(double num)
   {
      return 1.0 / (1.0 + Math.exp(-num));
   } // public static double f(double num)

   /**
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