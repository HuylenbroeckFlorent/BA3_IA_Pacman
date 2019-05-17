while true
do
  python3 pacman.py --frameTime 0.05 -p ReflexAgent
  c=$?
  [ $c -eq 0 ] || break
  python3 pacman.py -p ReflexAgent -l testClassic
  c=$?
  [ $c -eq 0 ] || break
  python3 pacman.py --frameTime 0.05 -p ReflexAgent -k 2
  c=$?
  [ $c -eq 0 ] || break
  python3 pacman.py --frameTime 0.05 -p ReflexAgent
  c=$?
  [ $c -eq 0 ] || break
  python3 pacman.py --frameTime 0.05 -p ReflexAgent -l openClassic
  c=$?
  [ $c -eq 0 ] || break
  python3 autograder.py -q q2
  c=$?
  [ $c -eq 0 ] || break
  python3 pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
  c=$?
  [ $c -eq 0 ] || break
  python3 pacman.py -p MinimaxAgent -l trappedClassic -a depth=3
  c=$?
  [ $c -eq 0 ] || break
  python3 pacman.py -p AlphaBetaAgent --frameTime 0.05 -a depth=3 -l smallClassic
  c=$?
  [ $c -eq 0 ] || break
  python3 pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
  c=$?
  [ $c -eq 0 ] || break
  python3 pacman.py -p AlphaBetaAgent -l trappedClassic -a depth=3
  c=$?
  [ $c -eq 0 ] || break
  python3 pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3
  c=$?
  [ $c -eq 0 ] || break
  python3 autograder.py -q q5
  c=$?
  [ $c -eq 0 ] || break
done

