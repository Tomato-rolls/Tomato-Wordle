"""
Tomato Wordle Solver

Usage:
    This script provides three modes for solving Wordle puzzles:

    1. Auto Mode (default):
       - Automatically guesses the target word using entropy-based strategy
       - Example: python solver.py --length 5 --target apple

    2. Interactive Mode:
       - Provides real-time suggestions based on your game feedback
       - Example: python solver.py --length 5 --interactive

    3. Manual Guess Mode:
       - Step-by-step guidance with entropy-ranked suggestions
       - Example: python solver.py --length 5 --manual-guess

    Common Arguments:
        --length       Word length (3-15), default=5
        --target       Specify target word for auto mode
        --interactive  Enable interactive feedback mode
        --manual-guess Enable manual guessing with suggestions

    The solver uses information theory to maximize information gain at each guess,
    significantly improving solving efficiency compared to random guessing.
"""

"""
Tomato Wordle Solver
Copyright (c) 2025 Tomato

Licensed under the MIT License.
"""

import math
import random
import argparse
import sys
from english_words import get_english_words_set
from tqdm import tqdm

MIN_LEN = 3
MAX_LEN = 15

def load_vocab():
    """Load vocabulary grouped by word length."""
    english_words_set = get_english_words_set(['web2'], lower=True)
    all_words = [w for w in english_words_set if w.isalpha()]
    vocab = {}
    for n in range(MIN_LEN, MAX_LEN + 1):
        vocab[n] = [w for w in all_words if len(w) == n][:500000]
    return vocab

VOCABULARY_BY_LENGTH = load_vocab()

def feedback(guess, target):
    """Generate feedback string ('g', 'y', 'b') for a guess against target."""
    fb = ['b'] * len(guess)
    used = [False] * len(target)
    for i in range(len(guess)):
        if guess[i] == target[i]:
            fb[i] = 'g'
            used[i] = True
    for i in range(len(guess)):
        if fb[i] == 'g':
            continue
        for j in range(len(target)):
            if not used[j] and guess[i] == target[j]:
                fb[i] = 'y'
                used[j] = True
                break
    return ''.join(fb)

def filter_candidates(candidates, guess, fb):
    """Filter candidates based on guess and feedback."""
    return [word for word in candidates if feedback(guess, word) == fb]

def calculate_entropy(guess, candidates):
    """Calculate entropy of a guess over candidate words."""
    total = len(candidates)
    pattern_counts = {}
    for target in candidates:
        f = feedback(guess, target)
        pattern_counts[f] = pattern_counts.get(f, 0) + 1
    entropy = 0
    for count in pattern_counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def rank_guesses(candidates, full_vocab, topn=100):
    """Rank guesses by information entropy and return top N."""
    entropies = []
    use_tqdm = sys.stdout.isatty()
    bar = tqdm(full_vocab, desc="Calculating entropy", leave=False, disable=not use_tqdm)
    for word in bar:
        e = calculate_entropy(word, candidates)
        entropies.append((word, e))
    entropies.sort(key=lambda x: -x[1])
    return entropies[:topn]

def visualize_candidates(turn, num_candidates, max_candidates, bar_width=40):
    """Visualize remaining candidates count with progress bar."""
    filled_len = int(bar_width * num_candidates / max_candidates)
    bar = '█' * filled_len + '-' * (bar_width - filled_len)
    print(f"[Turn {turn}] Remaining candidates: {num_candidates} |{bar}|")

def play_wordle(length=5, target=None, verbose=True):
    """Auto play Wordle simulation, returns number of turns used."""
    if length < MIN_LEN or length > MAX_LEN:
        raise ValueError(f"Only supports word length {MIN_LEN} to {MAX_LEN}.")
    full_vocab = VOCABULARY_BY_LENGTH[length]
    candidates = full_vocab.copy()
    max_candidates = len(candidates)

    if target:
        if len(target) != length or target not in full_vocab:
            print(f"❌ Target word '{target}' invalid or not in vocabulary.")
            return -1

    target = target or random.choice(candidates)
    turns = 0

    while True:
        turns += 1
        guess, ent = rank_guesses(candidates, candidates, 1)[0]
        fb = feedback(guess, target)
        if verbose:
            print(f"\nTurn {turns}: Guess = {guess}, Feedback = {fb}, Entropy = {ent:.2f} bits")
        if fb == 'g' * length:
            print(f"\n🎉 Guessed the word '{target}' in {turns} turns!")
            return turns
        candidates = filter_candidates(candidates, guess, fb)
        if not candidates:
            print("❌ No candidates left. Something went wrong.")
            return -1
        if verbose:
            visualize_candidates(turns, len(candidates), max_candidates)

def interactive_manual_mode(length):
    """Interactive mode where user inputs guesses and feedback."""
    full_vocab = VOCABULARY_BY_LENGTH.get(length)
    if not full_vocab:
        print("❌ No vocabulary available for this length.")
        return

    candidates = full_vocab.copy()
    turn = 0
    max_candidates = len(candidates)

    while True:
        turn += 1
        print(f"\nTurn {turn}: Current candidate count: {len(candidates)}")
        guess = input(f"Enter your guess (length {length}): ").strip().lower()
        if len(guess) != length or guess not in full_vocab:
            print("❌ Invalid word, please try again.")
            turn -= 1
            continue

        fb = input("Enter feedback (g=green, y=yellow, b=black; e.g. 'gybgb'): ").strip().lower()
        if len(fb) != length or any(c not in 'gyb' for c in fb):
            print("❌ Invalid feedback format, please try again.")
            turn -= 1
            continue

        if fb == 'g' * length:
            print(f"\n🎉 Success! Guessed '{guess}' in {turn} turns.")
            break

        candidates = filter_candidates(candidates, guess, fb)
        if not candidates:
            print("❌ No candidates match the criteria.")
            break

        visualize_candidates(turn, len(candidates), max_candidates)
        suggestions = rank_guesses(candidates, candidates, topn=100)
        print("Suggested guesses (ranked by entropy):")
        for i, (word, entropy) in enumerate(suggestions, 1):
            print(f"{i}. {word}  (Entropy: {entropy:.2f} bits)")

def main():
    print("""
        ████████╗ ██████╗ ███╗   ███╗ █████╗ ████████╗ ██████╗ 
        ╚══██╔══╝██╔═══██╗████╗ ████║██╔══██╗╚══██╔══╝██╔═══██╗
           ██║   ██║   ██║██╔████╔██║███████║   ██║   ██║   ██║
           ██║   ██║   ██║██║╚██╔╝██║██╔══██║   ██║   ██║   ██║
           ██║   ╚██████╔╝██║ ╚═╝ ██║██║  ██║   ██║   ╚██████╔╝
           ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ 
        """)
    print("=" * 40)

    parser = argparse.ArgumentParser(
        description="Information Entropy based Wordle Solver",
        usage=(
            "python %(prog)s [options]\n\n"
            "Examples:\n"
            "  Auto mode:       python %(prog)s --length 5 --target apple\n"
            "  Interactive:     python %(prog)s --length 5 --interactive\n"
            "  Manual guess:    python %(prog)s --length 5 --manual-guess"
        )
    )
    parser.add_argument('--length', type=int, default=5, help="Word length (3-15), default=5")
    parser.add_argument('--target', type=str, default=None, help="Target word for auto mode")
    parser.add_argument('--interactive', action='store_true', help="Enable interactive feedback mode")
    parser.add_argument('--manual-guess', action='store_true', help="Enable manual guessing in interactive mode")

    args = parser.parse_args()

    if args.interactive or args.manual_guess:
        interactive_manual_mode(args.length)
    else:
        play_wordle(length=args.length, target=args.target)

if __name__ == "__main__":
    main()
