package main

import (
    "encoding/json"
    "log"
    "net/http"
    "sync"
)

// User represents a user in the system
type User struct {
    ID       int    `json:"id"`
    Name     string `json:"name"`
    Email    string `json:"email"`
    IsActive bool   `json:"is_active"`
}

// UserStore manages user data with thread safety
type UserStore struct {
    mu    sync.RWMutex
    users map[int]User
}

// NewUserStore creates a new user store
func NewUserStore() *UserStore {
    return &UserStore{
        users: make(map[int]User),
    }
}

// Get retrieves a user by ID
func (s *UserStore) Get(id int) (User, bool) {
    s.mu.RLock()
    defer s.mu.RUnlock()
    user, ok := s.users[id]
    return user, ok
}

// Set stores a user
func (s *UserStore) Set(user User) {
    s.mu.Lock()
    defer s.mu.Unlock()
    s.users[user.ID] = user
}

// HandleGetUser handles GET /users/:id
func (s *UserStore) HandleGetUser(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func main() {
    store := NewUserStore()
    http.HandleFunc("/users/", store.HandleGetUser)
    log.Println("Server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
