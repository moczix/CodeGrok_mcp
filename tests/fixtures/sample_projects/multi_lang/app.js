/**
 * Main application entry point.
 */

class UserService {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
        this.cache = new Map();
    }

    async fetchUser(userId) {
        if (this.cache.has(userId)) {
            return this.cache.get(userId);
        }

        const response = await fetch(`${this.apiUrl}/users/${userId}`);
        const user = await response.json();
        this.cache.set(userId, user);
        return user;
    }

    async createUser(userData) {
        const response = await fetch(`${this.apiUrl}/users`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(userData)
        });
        return response.json();
    }
}

function formatDate(date) {
    return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    }).format(date);
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

module.exports = { UserService, formatDate, debounce };
