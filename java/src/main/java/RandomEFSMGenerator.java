

/*-
 * #%L
 * This file is part of ehW inference
 * Please visit http://achar.net/ for further information
 * 
 * Authors : Roland Groz, German Vega, Luca Devlin
 *           Laboratoire d'Informatique de Grenoble,University of Sheffield
 * %-
 * Copyright (C) 2020 - 2026 Achar group
 * %-
 * This program and the accompanying materials are made
 * available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 * 
 * SPDX-License-Identifier: EPL-2.0
 * #L%
 */


import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Random;
import java.util.Set;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import net.achar.efsm.Domain;
import static net.achar.efsm.Domain.*;
import net.achar.efsm.EFSM;
import net.achar.efsm.Parameterized;
import static net.achar.efsm.Parameterized.*;
import net.achar.learning.StatePartialInputSUL;
import net.achar.learning.StatePartialInputSUL.Observation;
import net.achar.serialization.dot.HtmlLabelVisualization;
import net.achar.serialization.dot.efsm.DotEFSM;
import net.achar.util.csv.Trace;
import net.achar.util.csv.Trace.Column;
import net.achar.util.efsm.EFSMConjecture.Sampled;
import static net.achar.util.efsm.EFSMSimulation.*;

/**
 * Generates a random EFSM based on Roland's algorithm.
 * Produces two files:
 * 1. A .dot file representing the EFSM's graph structure.
 * 2. A .java test file for sampling the generated .dot file.
 */
public class RandomEFSMGenerator {
    
    enum ParamType { INTEGER, FLOAT }

    static class Parameter {
        String name;
        ParamType type;
        Parameter(String name, ParamType type) {
            this.name = name;
            this.type = type;
        }
        @Override
        public String toString() { return name + ":" + (type == ParamType.INTEGER ? "int" : "float"); }
        String toJavaType() { return type == ParamType.INTEGER ? "Integer" : "Double"; }
        String toJavaValue(Random rand) {
            return type == ParamType.INTEGER ? 
                   Integer.toString(rand.nextInt(1000)) : 
                   String.format("%.2f", rand.nextDouble() * 1000);
        }
    }

    static class InputType {
        String name;
        List<Parameter> params;
        InputType(String name, List<Parameter> params) { this.name = name; this.params = params; }
    }

    static class OutputType {
        String name;
        List<Parameter> params;
        OutputType(String name, List<Parameter> params) { this.name = name; this.params = params; }
    }

    static class Expression {
        String expr;
        Expression(String expr) { this.expr = expr; }
        @Override
        public String toString() { return expr; }
    }

    static class Guard {
        String left;
        String operator; // "==", "<", ">" if complexity = 0
        String right;
        Guard(String left, String op, String right) {
            this.left = left; this.operator = op; this.right = right;
        }
        @Override
        public String toString() { return left + " " + operator + " " + right; }
        
        Guard negated() {
            String newLeft = this.left
                .replace("==", "__EQ__")
                .replace("!=", "__NEG__")
                .replace("<=", "__LE__")
                .replace(">=", "__GE__")
                .replace("<", "__LT__")
                .replace(">", "__GT__")
                .replace("and", "__AND__")
                .replace("or", "__OR__");
            newLeft = newLeft
                .replace("__EQ__", "!=")
                .replace("__NEG__", "==")
                .replace("__LE__", ">")
                .replace("__GE__", "<")
                .replace("__LT__", ">=")
                .replace("__GT__", "<=")
                .replace("__AND__", "or")
                .replace("__OR__", "and");

            String newOp = this.operator
                .replace("==", "__EQ__")
                .replace("!=", "__NEG__")
                .replace("<=", "__LE__")
                .replace(">=", "__GE__")
                .replace("<", "__LT__")
                .replace(">", "__GT__")
                .replace("and", "__AND__")
                .replace("or", "__OR__");
            newOp = newOp
                .replace("__EQ__", "!=")
                .replace("__NEG__", "==")
                .replace("__LE__", ">")
                .replace("__GE__", "<")
                .replace("__LT__", ">=")
                .replace("__GT__", "<=")
                .replace("__AND__", "or")
                .replace("__OR__", "and");

            String newRight = this.right
                .replace("==", "__EQ__")
                .replace("!=", "__NEG__")
                .replace("<=", "__LE__")
                .replace(">=", "__GE__")
                .replace("<", "__LT__")
                .replace(">", "__GT__")
                .replace("and", "__AND__")
                .replace("or", "__OR__");
            newRight = newRight
                .replace("__EQ__", "!=")
                .replace("__NEG__", "==")
                .replace("__LE__", ">")
                .replace("__GE__", "<")
                .replace("__LT__", ">=")
                .replace("__GT__", "<=")
                .replace("__AND__", "or")
                .replace("__OR__", "and");
            return new Guard(newLeft, newOp, newRight);
        }
    }

    static class Transition {
        State source;
        State target;
        InputType input;
        Guard guard;
        String outputName;
        List<Expression> outputAssignments; // One per output param
        List<Expression> registerUpdates;

        Transition(State source, State target, InputType input, String outputName, List<Expression> assignments) {
            this.source = source;
            this.target = target;
            this.input = input;
            this.outputName = outputName;
            this.outputAssignments = assignments;
            this.registerUpdates = new ArrayList<>(); // Qdded after
        }
    }

    static class State {
        String name;
        List<Transition> transitions = new ArrayList<>();
        State(String name) { this.name = name; }
        @Override
        public boolean equals(Object obj) { return obj instanceof State && ((State)obj).name.equals(this.name); }
    }

    private final Random r(and;
    private List<InputType> inputTypes;
    private List<OutputType> outputTypes;
    private final Map<String, OutputType> outputTypeMap = new HashMap<>();
    private List<String> registers; // internal registers r0, r1..., and input+output registers i0, i1..., o0, o1...
    private List<ParamType> registerTypes; // The dataTypes of the registers r0, r1...
    private List<String> internalRegisters;
    private List<State> states;
    private State initialState;
    private boolean paramCreated = false;
    
    /**
     * Main generation method.
     * @param numInputTypes Number of input types (2-5)
     * @param numOutputTypes Number of output types (2-4)
     * @param numStates Number of states (3-12)
     * @param maxInputParams Max params for inputs (0-2)
     * @param maxOutputParams Max params for outputs (0-2)
     * @param numGuardedTransitions Number of guarded transitions to add (2-4)
     * @param numInternalRegisters Number of internal registers (e.g. 3 for r0,r1,r2)
     * @return The generated EFSM as a .dot file string.
     */
    public String generate(int numInputTypes, int numOutputTypes, int numStates, 
                           int maxInputParams, int maxOutputParams, int numGuardedTransitions,
                           int numInternalRegisters, int complexity) {

        // 2.5: Registers
        this.registers = new ArrayList<>();
        this.registerTypes = new ArrayList<>();
        this.internalRegisters = new ArrayList<>();
        for (int i = 0; i < numInternalRegisters; i++) {
            // For simplicity, all registers are integers
            this.registerTypes.add(ParamType.INTEGER);
            this.registers.add("r" + i);
            this.internalRegisters.add("r" + i);
        }
        
        // 1: Input Types
        this.inputTypes = new ArrayList<>();
        for (int i = 0; i < numInputTypes; i++) {
            this.inputTypes.add(generateInputType(i, maxInputParams));
        }
        
        //2: Output Types
        this.outputTypes = new ArrayList<>();
        this.outputTypeMap.clear();
        for (int i = 0; i < numOutputTypes; i++) {
            OutputType ot = generateOutputType(i, maxOutputParams, numOutputTypes);
            this.outputTypes.add(ot);
            this.outputTypeMap.put(ot.name, ot);
        }

        // 3: States
        this.states = new ArrayList<>();
        this.initialState = new State("s0");
        this.states.add(initialState);
        for (int i = 1; i < numStates; i++) {
            this.states.add(new State("s" + i));
        }

        // Step 4 + 5: Graph (BFS)
        // Ensure each state has one transition for each input type
        Queue<State> stateQueue = new LinkedList<>(this.states);
        Set<State> processed = new HashSet<>();

        stateQueue.remove(initialState);
        processed.add(initialState);
        processState(initialState, numStates, complexity);
        
        while (!stateQueue.isEmpty()) {
            State currentState = stateQueue.poll();
            if (!processed.contains(currentState)) {
                processState(currentState, numStates, complexity);
                processed.add(currentState);
            }
        }

        // Step 6: Add Guards
        addGuardedTransitions(numGuardedTransitions, complexity);

        // Step 7: Add reset to each output at the latest possible state
        Map<String, Transition> lastTransitionAddedUsingOutputMap = new HashMap<>();

        for (State s: this.states) {
            for (Transition t: s.transitions) {
                if (!t.outputName.equals(OMEGA)) {
                    // gets updated with a new transition for every use - so will give the last transition that uses that output
                    lastTransitionAddedUsingOutputMap.put(t.outputName, t);
                }
            }
        }

        for (Transition t: lastTransitionAddedUsingOutputMap.values()) {
            OutputType ot = outputTypeMap.get(t.outputName);

            if (ot != null) {
                List<Expression> zeroAssignments = new ArrayList<>();
                for (int i=0; i < ot.params.size(); i++) {
                    zeroAssignments.add(new Expression("0"));
                }

                t.outputAssignments = zeroAssignments;
            }
        }

        // --- Check Connectivity ---
        if (!isStronglyConnected()) {
            return "";
        }

        return toDotString();
    }

    /**
     * Populate states aith transitions
     */
    private void processState(State currentState, int numStates, int complexity) {
        for (InputType input : inputTypes) {
            // Pick output (or OMEGA)
            String outputName = pickRandom(outputTypes.stream().map(o -> o.name).collect(Collectors.toList()));
            
            State targetState;
            if (outputName.equals(OMEGA)) {
                targetState = currentState; // Self-loop
            } else {
                targetState = pickRandom(states);
            }
            
            // Step 5: Output Functions
            List<String> Allowed = new ArrayList<>(registers);
            for (Parameter param : input.params) {
                Allowed.add(param.name);
            }

            List<Expression> assignments = createOutputAssignments(outputName, Allowed, complexity);

            OutputType ot = outputTypeMap.get(outputName);

            if (ot != null && ot.params != null) {
                for (Parameter param : ot.params) {
                    Allowed.add(param.name);
                }
            }
            
            Transition t = new Transition(currentState, targetState, input, outputName, assignments);
            t.registerUpdates = createGlobalRegisterUpdates(Allowed, input, ot); // Finally qdd register updates
            currentState.transitions.add(t);
        }
    }

    /**
     * Add guards to existing trqnsitions
     */
    private void addGuardedTransitions(int numGuards, int complexity) {
        List<Transition> nonOmegaTransitions = new ArrayList<>();
        for (State s : states) {
            for (Transition t : s.transitions) {
                if (!t.outputName.equals(OMEGA)) {
                    nonOmegaTransitions.add(t);
                }
            }
        }
        
        if (nonOmegaTransitions.isEmpty()) {
            System.err.println("No non-OMEGA transitions found to add guards to.");
            return;
        }

        for (int i = 0; i < numGuards && !nonOmegaTransitions.isEmpty(); i++) {
            Transition t = pickRandom(nonOmegaTransitions);
            nonOmegaTransitions.remove(t);

            List<String> Allowed = new ArrayList<>(registers);
            InputType input = t.input;
            for (Parameter param : input.params) {
                Allowed.add(param.name);
            }

            Guard guard = createRandomGuard(Allowed, complexity);
            t.guard = guard;

            // Create new transition with negation
            Guard negatedGuard = guard.negated();
            
            // Pick a different output
            List<String> otherOutputs = outputTypes.stream()
                .map(o -> o.name)
                .filter(name -> !name.equals(t.outputName))
                .collect(Collectors.toList());
            String newOutputName = pickRandom(otherOutputs);

            State newTarget;
            if (newOutputName.equals(OMEGA)) {
                newTarget = t.source;
            } else {
                newTarget = pickRandom(states);
            }

            List<Expression> newAssignments = createOutputAssignments(newOutputName, Allowed, complexity);

            OutputType ot = outputTypeMap.get(newOutputName);

            if (ot != null && ot.params != null) {
                for (Parameter param : ot.params) {
                    Allowed.add(param.name);
                }
            }
            
            Transition guardedTransition = new Transition(t.source, newTarget, t.input, newOutputName, newAssignments);
            guardedTransition.guard = negatedGuard;
            guardedTransition.registerUpdates = createGlobalRegisterUpdates(Allowed, t.input, ot);
            
            t.source.transitions.add(guardedTransition);
        }
    }

    /**
     * Creates a random guard
     */
    private Guard createRandomGuard(List<String> available, int complexity) {
        
        if (complexity == 1) {
            String left = (new Expression(pickRandom(available))).toString();
            String op = pickRandom(guardOps);
            String right = rand.nextBoolean() ? 
                (new Expression(pickRandom(available))).toString() : 
                createRandomConstant().toString();
                return new Guard(left, op, right);
        } else if (complexity == 2) {
            String left = "(" + createRandomGuard(available, 1).toString() + ")";
            String op = pickRandom(List.of("and", "or"));
            String right = "(" + createRandomGuard(available, 1).toString() + ")";
            return new Guard(left, op, right);
        } else {
            String left = "(" + createRandomGuard(available, 2).toString() + ")";
            String op = pickRandom(List.of("and", "or"));
            String right = "(" + createRandomGuard(available, 1).toString() + ")";
            return new Guard(left, op, right);
        }    
    }

    /**
     * Creates a random update for registers. e.g., (r0, r1 + 1, i0)
     */
    private List<Expression> createGlobalRegisterUpdates(List<String> available, InputType input, OutputType output) {
        List<Expression> updates = new ArrayList<>();
        for (int i = 0; i < this.internalRegisters.size(); i++) {
            // 50% chance to keep the old value (e.g., "r0")
            if (rand.nextInt(2) == 0) {
                updates.add(new Expression("r" + i));
            } else {
                // 50% chance to generate a new expression
                updates.add(createRandomExpression(available, 1));
            }
        }

        for (int i=0; i < this.registers.size(); i++) {
            if (this.registers.get(i).startsWith(input.name)) {
                for (int j = 0; j < input.params.size(); j++) {
                    if (this.registers.get(i).contains(input.params.get(j).name)) {
                        updates.add(new Expression(input.params.get(j).name));
                    }
                }
            } else if (output != null) {
                if (this.registers.get(i).startsWith(output.name)) {
                    for (int j = 0; j < output.params.size(); j++) {
                        if (this.registers.get(i).contains(output.params.get(j).name)) {
                            updates.add(new Expression(output.params.get(j).name));
                        }
                    }
                } else if (this.registers.get(i).startsWith("i")) {
                    updates.add(new Expression(registers.get(i)));
                } else if (this.registers.get(i).startsWith("o")) {
                    updates.add(new Expression(registers.get(i)));
                }
            } else if (this.registers.get(i).startsWith("i")) {
                updates.add(new Expression(registers.get(i)));
            } else if (this.registers.get(i).startsWith("o")) {
                updates.add(new Expression(registers.get(i)));
            }
        }
        return updates;
    }

    /**
     * Creates a random output function
     */
    private List<Expression> createOutputAssignments(String outputName, List<String> available, int complexity) {
        if (outputName.equals(OMEGA) || !outputTypeMap.containsKey(outputName)) {
            return Collections.emptyList();
        }
        
        List<Expression> assignments = new ArrayList<>();
        OutputType ot = outputTypeMap.get(outputName);
        for (Parameter p : ot.params) {
            assignments.add(createRandomExpression(available, complexity));
        }
        return assignments;
    }

    /**
     * Creates a random expression
     * Note: "registers" contains "r0", "i0", "o0", etc.
     */
    private Expression createRandomExpression(List<String> available, int complexity) {
        if (available.isEmpty()) {
            return createRandomConstant(); // Fallback if no registers defined
        }

        if (complexity == 2) {
            return new Expression(createRandomExpression(available, 1).toString() + " " + pickRandom(arithOps) + " " + pickRandom(available));
        } else if (complexity == 3) {
            return new Expression(createRandomExpression(available, 2).toString() + " " + pickRandom(arithOps) + " " + pickRandom(available));
        }
        
        int choice = rand.nextInt(4);
        switch (choice) {
            case 0: // register
                return new Expression(pickRandom(available));
            case 1: // constant
                return createRandomConstant();
            case 2: // register op register
                return new Expression(pickRandom(available) + " " + pickRandom(arithOps) + " " + pickRandom(available));
            case 3: // register op constant
                return new Expression(pickRandom(available) + " " + pickRandom(arithOps) + " " + createRandomConstant().expr);
            default:
                return new Expression(pickRandom(available));
        }
    }

    private Expression createRandomConstant() {
        // Only integer constants
        return new Expression(Integer.toString(rand.nextInt(10) + 1));
    }
    
    /**
     * Creates a random inputType
     */
    private InputType generateInputType(int index, int maxParams) {
        String name = "in" + index;
        int numParams = rand.nextInt(maxParams + 1);
        if (numParams > 0) {
            paramCreated = true;
        }

        List<Parameter> params = new ArrayList<>();
        for (int i = 0; i < numParams; i++) {
            String pName = "p" + i;
            String iName = "i" + i; // Use i0, i1 for expressions
            // For simplicity, let's use Integers only for now
            Parameter p = new Parameter(iName, ParamType.INTEGER);
            params.add(p);
            this.registerTypes.add(p.type);
            this.registers.add(name + iName);
        }
        return new InputType(name, params);
    }
    
    /**
     * Creates a random outputType
     */
    private OutputType generateOutputType(int index, int maxParams, int numOutputTypes) {
        String name = "out" + index;
        int numParams = rand.nextInt(maxParams + 1);
        if (numParams > 0) {
            paramCreated = true;
        }
        if ((index == numOutputTypes - 1) && (paramCreated == false)) {
            numParams ++;
        }

        List<Parameter> params = new ArrayList<>();
        for (int i = 0; i < numParams; i++) {
            String pName = "v" + i;
            String oName = "o" + i; // Use o0, o1 for expressions
            Parameter p = new Parameter(oName, ParamType.INTEGER);
            params.add(p);
            this.registerTypes.add(p.type);
            this.registers.add(name + oName);
        }
        return new OutputType(name, params);
    }

    // --- Utility ---

    private <T> T pickRandom(List<T> list) {
        return list.get(rand.nextInt(list.size()));
    }

    private String pickRandom(List<String> list, String extraOption) {
        if (rand.nextBoolean() || list.isEmpty()) {
            return extraOption;
        }
        return pickRandom(list);
    }

    /**
     * Checks if the EFSM is strongly connected
     */
    private boolean isStronglyConnected() {
        if (states.isEmpty()) return true;
        
        // Check all nodes reachable from start
        Set<State> visited = new HashSet<>();
        dfs(initialState, visited, false);
        if (visited.size() != states.size()) {
            return false;
        }

        // Check all nodes can reach start (by checking reachability on reversed graph)
        visited.clear();
        dfs(initialState, visited, true);
        return visited.size() == states.size();
    }

    /**
     * Depth first search helper for checking connectivity
     */
    private void dfs(State u, Set<State> visited, boolean reverse) {
        visited.add(u);
        if (reverse) {
            for (State v : states) {
                for (Transition t : v.transitions) {
                    if (t.target.equals(u) && !visited.contains(v)) {
                        dfs(v, visited, true);
                    }
                }
            }
        } else {
            for (Transition t : u.transitions) {
                if (!visited.contains(t.target)) {
                    dfs(t.target, visited, false);
                }
            }
        }
    }
    
    /**
     * Converts the generated EFSM to a .dot file string in our custom format
     */
    public String toDotString() {
        StringBuilder dot = new StringBuilder("digraph g {\n\n");
        dot.append("    /* Randomly Generated EFSM */\n\n");

        // Registers
        StringBuilder regString = new StringBuilder();
        for (int i=0; i<registers.size(); i++) {
            String registerName = registers.get(i);
            regString.append(registerName + ":");
            String registerType = switch (registerTypes.get(i)) {
                case INTEGER -> "INTEGER";
                case FLOAT -> "DECIMAL"; 
                default -> "UNKOWN";               
            };
            regString.append(registerType);
            if (i != registers.size() - 1) {
                regString.append(",");
            }

        }
        dot.append("    registers=\"").append(regString.toString()).append("\"\n\n");

        // Inputs
        List<String> inputDefs = new ArrayList<>();
        for (InputType in : inputTypes) {
            String p = in.params.stream()
                .map(param -> (param.type == ParamType.INTEGER ? "INTEGER" : "DECIMAL"))
                .collect(Collectors.joining(","));
            if (p.isEmpty()) {
                inputDefs.add(in.name);
            } else {
                inputDefs.add(in.name + "(" + p + ")");
            }
        }
        dot.append("    inputs=\"").append(String.join(" | ", inputDefs)).append("\"\n\n");

        // Outputs
        List<String> outputDefs = new ArrayList<>();
        for (OutputType out : outputTypes) {
             String p = out.params.stream()
                .map(param -> (param.type == ParamType.INTEGER ? "INTEGER" : "DECIMAL"))
                .collect(Collectors.joining(","));
            if (p.isEmpty()) {
                outputDefs.add(out.name);
            } else {
                outputDefs.add(out.name + "(" + p + ")");
            }
        }
        dot.append("    outputs=\"").append(String.join(" | ", outputDefs)).append("\"\n\n");

        //States
        for (State s : states) {
            String shape = s.equals(initialState) ? "box" : "circle";
            dot.append(String.format("    %s [shape=\"%s\" label=\"%s\"];\n", s.name, shape, s.name));
        }
        dot.append("\n");

        //Transition Definitions
        for (State s : states) {
            for (Transition t : s.transitions) {
                // Input
                String inputLabel = t.input.name;
                
                // [Guard]
                String guardLabel = (t.guard == null) ? "" : "[" + t.guard.toString() + "]";
                
                // / Output(assignments)
                String outputLabel;
                if (t.outputName.equals(OMEGA)) {
                    outputLabel = " / " + OMEGA + "()";
                } else {
                    String assignmentStr = t.outputAssignments.stream()
                        .map(Expression::toString)
                        .collect(Collectors.joining(","));
                    outputLabel = " / " + t.outputName + "(" + assignmentStr + ")";
                }

                // (Register Updates)
                String regUpdateStr = t.registerUpdates.stream()
                    .map(Expression::toString)
                    .collect(Collectors.joining(", "));
                String registerUpdateLabel = " (" + regUpdateStr + ")";

                // Our format
                String htmlLabel = (inputLabel + guardLabel + outputLabel + registerUpdateLabel)
                                    .replace("<", "&lt;")
                                    .replace(">", "&gt;")
                                    .replace("==", "=")
                                    .replace("!=", "not =");

                dot.append(String.format("    %s -> %s [label=<%s>];\n",
                    t.source.name, t.target.name, htmlLabel));
            }
        }

        // Start Node
        dot.append("\n");
        dot.append("    __start0 [label=\"\", shape=\"none\", width=\"0\", height=\"0\"];\n");
        dot.append(String.format("    __start0 -> %s;\n", initialState.name));

        dot.append("}\n");
        String original = dot.toString();
        String copy = original;

        for (int i = 0; i < registers.size(); i++) {
            String originalName = registers.get(i);
            String newName = "r" + i;

            copy = copy.replace(originalName, newName);

        }

        return copy;
    }

    public RandomEFSMGenerator(long seed) {
        this.rand = new Random(seed);
    }

    
    /**
     * Main method to run the generator.
     */
    public static void main(String[] args) throws Exception  {
        if (args.length < 11) {
            System.out.println("Missing arguments, format: ");
            System.out.println("NUM_INPUT_TYPES NUM_OUTPUT_TYPES NUM_STATES MAX_INPUT_PARAMS MAX_OUTPUT_PARAMS NUM_GUARDED_TRANSITIONS NUM_INTERNAL_REGISTERS NUM_EFSMS_TO_GENERATE NUM_SAMPLES COMPLEXITY ?WHITE_BOX? [SEED]");
            return;
        }
    
        // --- Configuration ---
        int NUM_INPUT_TYPES = Integer.parseInt(args[0]);
        int NUM_OUTPUT_TYPES = Integer.parseInt(args[1]);
        int NUM_STATES = Integer.parseInt(args[2]);   // fixed typo
        int MAX_INPUT_PARAMS = Integer.parseInt(args[3]);
        int MAX_OUTPUT_PARAMS = Integer.parseInt(args[4]);
        int NUM_GUARDED_TRANSITIONS = Integer.parseInt(args[5]);
        int NUM_INTERNAL_REGISTERS = Integer.parseInt(args[6]);
        int NUM_EFSMS = Integer.parseInt(args[7]);
        int NUM_SAMPLES = Integer.parseInt(args[8]);
        int COMPLEXITY = Integer.parseInt(args[9]);
        int WHITE_BOX = Integer.parseInt(args[10]);
    
        long SEED = (args.length >= 12)
                ? Long.parseLong(args[11])
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
                
                while (dotContent.equals("")) {
                    dotContent = generator.generate(
                    NUM_INPUT_TYPES, NUM_OUTPUT_TYPES, NUM_STATES,
                    MAX_INPUT_PARAMS, MAX_OUTPUT_PARAMS, NUM_GUARDED_TRANSITIONS,
                    NUM_INTERNAL_REGISTERS, COMPLEXITY);
                }

                String dotFileName = baseDotName + "_" + i + ".dot";
                Path dotPath = Path.of(dotFileName);
                Files.writeString(dotPath, dotContent);
                System.out.println("Successfully generated EFSM model to: " + dotPath.toAbsolutePath());

            } catch (IOException e) {
                System.err.println("Error generating files: " + e.getMessage());
            }
        }
    }

    private static <S,R> Column.Accessor<EFSM.Configuration<S,R>,?,?,R> fromTargetRegister() {
		return Trace.<EFSM.Configuration<S,R>,Object,Object> fromTarget().then(EFSM.Configuration::register);
	}

    private static <R> Trace log(String resource, Domain<R> signature, String delimiter) throws IOException {
		return Trace.trace().withDelimiter(delimiter)
				.withDefaultColumns()
				.withColumns(Trace.split(Trace.column("r",fromTargetRegister()),signature))
				.to(Path.of(resource));
	}

	private static <R> Trace log(String resource, Domain<R> signature) throws IOException {
		return log(resource,signature,",");
	}

   
	private static <I,O> Stream<I> walk(StatePartialInputSUL<I,O> sul, RandomGenerator generator, int limit, List<I> samples) {

		return Stream.generate(() -> generator.nextInt(samples.size()))
					.limit(limit)
					.map(index -> samples.get(index))
					.sequential().filter(input -> sul.isDefinedInput(input));
	}

}
