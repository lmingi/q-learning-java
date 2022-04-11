
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class QLearning {

    public double alpha = 0.1; // Learning rate
    private final double gamma = 0.9; // Eagerness - 0 looks in the near future, 1 looks in the distant future
    public double epsilon = 0.25; // Eagerness - 0 looks in the near future, 1 looks in the distant future

    private final int mazeWidth = 4;
    private final int mazeHeight = 4;
    private final int statesCount = mazeHeight * mazeWidth;

    private final int reward = 1;

    private char[][] maze; // Maze read from file
    private int[][] R; // Reward lookup
    private double[][] Q; // Q learning

    public static void main(String args[]) {
        QLearning ql = new QLearning();

        ql.init();
        ql.calculateQ();
        ql.printQ();
        ql.printPolicy();
    }

    public void init() {
        File file = new File("/Users/leemingi/Documents/CS/q-learning-java/src/main/java/com/technobium/rl/maze.txt");

        R = new int[statesCount][statesCount];
        Q = new double[statesCount][statesCount];
        maze = new char[mazeHeight][mazeWidth];

        try (FileInputStream fis = new FileInputStream(file)) {

            int i = 0;
            int j = 0;

            int content;

            // Read the maze from the input file
            while ((content = fis.read()) != -1) {
                char c = (char) content;
                if (c != '0' && c != 'F' && c != 'X') {
                    continue;
                }
                maze[i][j] = c;
                j++;
                if (j == mazeWidth) {
                    j = 0;
                    i++;
                }
            }

            // We will navigate through the reward matrix R using k index
            for (int k = 0; k < statesCount; k++) {

                // We will navigate with i and j through the maze, so we need
                // to translate k into i and j
                i = k / mazeWidth;
                j = k - i * mazeWidth;

                // Fill in the reward matrix with 0
                for (int s = 0; s < statesCount; s++) {
                    R[k][s] = 0;
                }

                // If not in final state or a wall try moving in all directions in the maze
                if (maze[i][j] != 'F') {

                    // Try to move left in the maze
                    int goLeft = j - 1;
                    if (goLeft >= 0) {
                        int target = i * mazeWidth + goLeft;
                        if (maze[i][goLeft] == '0') {
                            R[k][target] = -1;
                        } else if (maze[i][goLeft] == 'F') {
                            R[k][target] = reward;
                        } else {
                            R[k][target] = -1;
                        }
                    }

                    // Try to move right in the maze
                    int goRight = j + 1;
                    if (goRight < mazeWidth) {
                        int target = i * mazeWidth + goRight;
                        if (maze[i][goRight] == '0') {
                            R[k][target] = -1;
                        } else if (maze[i][goRight] == 'F') {
                            R[k][target] = reward;
                        } else {
                            R[k][target] = -1;
                        }
                    }

                    // Try to move up in the maze
                    int goUp = i - 1;
                    if (goUp >= 0) {
                        int target = goUp * mazeWidth + j;
                        if (maze[goUp][j] == '0') {
                            R[k][target] = -1;
                        } else if (maze[goUp][j] == 'F') {
                            R[k][target] = reward;
                        } else {
                            R[k][target] = -1;
                        }
                    }

                    // Try to move down in the maze
                    int goDown = i + 1;
                    if (goDown < mazeHeight) {
                        int target = goDown * mazeWidth + j;
                        if (maze[goDown][j] == '0') {
                            R[k][target] = -1;
                        } else if (maze[goDown][j] == 'F') {
                            R[k][target] = reward;
                        } else {
                            R[k][target] = -1;
                        }
                    }
                }
            }
            initializeQ();
            printR(R);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Set Q values to R values
    void initializeQ() {
        for (int i = 0; i < statesCount; i++) {
            for (int j = 0; j < statesCount; j++) {
                Q[i][j] = 0;
            }
        }
    }

    // Used for debug
    void printR(int[][] matrix) {
        System.out.printf("%25s", "States: ");
        for (int i = 0; i <= 16; i++) {
            System.out.printf("%4s", i);
        }
        System.out.println();

        for (int i = 0; i < statesCount; i++) {
            System.out.print("Possible states from " + i + " :[");
            for (int j = 0; j < statesCount; j++) {
                System.out.printf("%4s", matrix[i][j]);
            }
            System.out.println("]");
        }
    }

    void calculateQ() {
        Random rand = new Random();
        for (int j = 0; j < 10; j++) {
            System.out.println("The total reward: for episode " + (1 + j));
            initializeQ();
            double iteration = 1;
            for (int i = 0; i < 100; i++) { // Train cycles
                // Select random initial state
                int crtState = 12;
                int totalReward = 0;
                while (!isFinalState(crtState)) {
                    int[] actionsFromCurrentState = possibleActionsFromState(crtState);
                    int nextState;
                    double pos = rand.nextDouble();
                    if (pos >= epsilon) {
                        nextState = getPolicyFromState(crtState);
                    } else {
                        int index = rand.nextInt(actionsFromCurrentState.length);
                        nextState = actionsFromCurrentState[index];
                    }

                    double q = Q[crtState][nextState];
                    double maxQ = maxQ(nextState);
                    int r = R[crtState][nextState];

                    double value = q + alpha * (r + gamma * maxQ - q);
                    Q[crtState][nextState] = value;
                    totalReward += r;
                    if (crtState == nextState) {
                        totalReward -= 1;
                    }
                    crtState = nextState;
                    iteration++;
                }
                System.out.print(totalReward + " ");
                // System.out.print(iteration + " ");
                // System.out.printf("%.2f ", (totalReward / (double) iteration));
            }
            System.out.println();
        }
    }

    boolean isFinalState(int state) {
        int i = state / mazeWidth;
        int j = state - i * mazeWidth;

        return maze[i][j] == 'F';
    }

    int[] possibleActionsFromState(int state) {
        ArrayList<Integer> result = new ArrayList<>();
        for (int i = 0; i < statesCount; i++) {
            if (R[state][i] != 0) {
                result.add(i);
            }
        }

        // if corner
        if (state == 3 || state == 12) {
            result.add(state);
            result.add(state);
        }
        if (state == 1 || state == 2 || state == 4 || state == 8 || state == 13 || state == 14 || state == 7
                || state == 11) {
            result.add(state);

        }

        return result.stream().mapToInt(i -> i).toArray();
    }

    double maxQ(int nextState) {
        int[] actionsFromState = possibleActionsFromState(nextState);
        // the learning rate and eagerness will keep the W value above the lowest reward
        double maxValue = 0;
        for (int nextAction : actionsFromState) {
            double value = Q[nextState][nextAction];

            maxValue = Math.max(value, maxValue);
        }
        return maxValue;
    }

    void printPolicy() {
        System.out.println("\nPrint policy");
        for (int i = 0; i < statesCount; i++) {
            System.out.println("From state " + i + " goto state " + getPolicyFromState(i));
        }
    }

    int getPolicyFromState(int state) {
        int[] actionsFromState = possibleActionsFromState(state);

        double maxValue = -100;
        int policyGotoState = state;

        // Pick to move to the state that has the maximum Q value
        for (int nextState : actionsFromState) {
            if (nextState != state) {
                double value = Q[state][nextState];

                if (value > maxValue) {
                    maxValue = value;
                    policyGotoState = nextState;
                }
            }
        }
        return policyGotoState;
    }

    void printQ() {
        System.out.println("Q matrix");
        for (int i = 0; i < Q.length; i++) {
            System.out.print("From state " + i + ":  ");
            for (int j = 0; j < Q[i].length; j++) {
                System.out.printf("%6.2f ", (Q[i][j]));
            }
            System.out.println();
        }
    }
}
