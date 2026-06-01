import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

public class Main {
    public static void main(String[] args) throws Exception {
        if (args[0].equals("G")) {

            String[] params = Arrays.copyOfRange(args, 1, args.length);
            if (params.length < 10) {
                System.out.println("Missing arguments, format: ");
                System.out.println("NUM_INPUT_TYPES NUM_OUTPUT_TYPES NUM_STATES MAX_INPUT_PARAMS MAX_OUTPUT_PARAMS NUM_GUARDED_TRANSITIONS NUM_INTERNAL_REGISTERS NUM_EFSMS_TO_GENERATE NUM_SAMPLES COMPLEXITY ?WHITE_BOX? [SEED]");
                return;
            }
            int NUM_INPUT_TYPES = Integer.parseInt(params[0]);
            int NUM_OUTPUT_TYPES = Integer.parseInt(params[1]);
            int NUM_STATES = Integer.parseInt(params[2]);   // fixed typo
            int MAX_INPUT_PARAMS = Integer.parseInt(params[3]);
            int MAX_OUTPUT_PARAMS = Integer.parseInt(params[4]);
            int NUM_GUARDED_TRANSITIONS = Integer.parseInt(params[5]);
            int NUM_INTERNAL_REGISTERS = Integer.parseInt(params[6]);
            int NUM_EFSMS = Integer.parseInt(params[7]);
            int COMPLEXITY = Integer.parseInt(params[8]);
            int WHITE_BOX = Integer.parseInt(params[9]);

            long SEED = (args.length >= 11)
                    ? Long.parseLong(args[10])
                    : 0L;
            String baseDotName = "../generated_data/RandomEFSM";

            Random seedGen = new Random(SEED);

            for (int i = 1; i <= NUM_EFSMS; i++) {
                long seed = seedGen.nextLong();
                RandomEFSMGenerator generator = new RandomEFSMGenerator(seed * seed);

                try {
                    String dotContent = generator.generate(
                            NUM_INPUT_TYPES, NUM_OUTPUT_TYPES, NUM_STATES,
                            MAX_INPUT_PARAMS, MAX_OUTPUT_PARAMS, NUM_GUARDED_TRANSITIONS,
                            NUM_INTERNAL_REGISTERS, COMPLEXITY);

                    while (dotContent.isEmpty()) {
                        dotContent = generator.generate(
                                NUM_INPUT_TYPES, NUM_OUTPUT_TYPES, NUM_STATES,
                                MAX_INPUT_PARAMS, MAX_OUTPUT_PARAMS, NUM_GUARDED_TRANSITIONS,
                                NUM_INTERNAL_REGISTERS, COMPLEXITY);
                    }

                    String dotFileName = baseDotName + "_" + i + ".dot";
                    Path dotPath = Path.of(dotFileName);
                    Files.writeString(dotPath, dotContent);
                    System.out.println("Successfully generated EFSM model to: " + dotPath.toAbsolutePath());

                    String pngFileName = baseDotName + "_" + i + ".png";
                    Process process = new ProcessBuilder("dot", "-Tpng", dotFileName, "-o", pngFileName)
                            .redirectErrorStream(true)
                            .start();
                    String processOutput = new String(process.getInputStream().readAllBytes());
                    int exitCode = process.waitFor();
                    if (exitCode != 0) {
                        System.err.println("Error generating PNG: " + processOutput);
                    } else {
                        System.out.println("Successfully generated PNG: " + pngFileName);
                    }

                } catch (IOException e) {
                    System.err.println("Error generating files: " + e.getMessage());
                }
            }

            return;
        }
        if (args[0].equals("S")) {
            System.out.println("Sampling mode selected.");
        /*
        // Call the Sampler class with the provided arguments
        String[] params = Arrays.copyOfRange(args, 1, args.length);
        if (params.length < 2) {
            System.out.println("Missing arguments, format: ");
            System.out.println("DOT_FILE NUM_SAMPLES SEED");
            return;
        }
        String dotFile = params[0];
        int numSamples = Integer.parseInt(params[1]);
        long SEED = (args.length >= 12)
                ? Long.parseLong(args[11])
                : 0L;
        Sampler sampler = new Sampler();
        sampler.sample(dotFile, numSamples, seed);

         */
        } else {
            System.out.println("Invalid mode. Use 'G' for generation or 'S' for sampling.");
        }
    }
}
