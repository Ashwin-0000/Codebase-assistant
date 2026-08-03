/**
 * Sample JavaScript module for testing the ingestion pipeline.
 *
 * Covers:
 *   - Named function declaration with JSDoc
 *   - Arrow function assigned to const
 *   - Class with constructor and methods
 *   - Nested function
 */

// Named function declaration
/**
 * Multiply two numbers together.
 * @param {number} a
 * @param {number} b
 * @returns {number}
 */
function multiply(a, b) {
  return a * b;
}

// Arrow function — name comes from the variable declaration
const divide = (a, b) => {
  if (b === 0) throw new Error("Division by zero");
  return a / b;
};

// Class with methods
class EventEmitter {
  constructor() {
    /** @type {Map<string, Function[]>} */
    this._listeners = new Map();
  }

  /**
   * Register a listener for the given event.
   * @param {string} event
   * @param {Function} listener
   */
  on(event, listener) {
    if (!this._listeners.has(event)) {
      this._listeners.set(event, []);
    }
    this._listeners.get(event).push(listener);
    return this;
  }

  /**
   * Emit an event, calling all registered listeners.
   * @param {string} event
   * @param {...any} args
   */
  emit(event, ...args) {
    const listeners = this._listeners.get(event) || [];
    listeners.forEach(fn => fn(...args));
  }

  // No JSDoc — should be noted by chunker
  off(event, listener) {
    const existing = this._listeners.get(event) || [];
    this._listeners.set(event, existing.filter(fn => fn !== listener));
  }
}

/**
 * Process a list of events using the multiply function.
 * @param {Array<{a: number, b: number}>} events
 * @returns {number[]}
 */
function processEvents(events) {
  const emitter = new EventEmitter();
  const results = events.map(e => multiply(e.a, e.b));
  emitter.emit('done', results);
  return results;
}

module.exports = { multiply, divide, EventEmitter, processEvents };
