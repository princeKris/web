import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, catchError, tap, throwError } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class Auth {
  private apiUrl = "http://localhost:8000/api";

  private userSubject = new BehaviorSubject<any>(null);
  user$ = this.userSubject.asObservable();

  constructor(private http: HttpClient,) {
    const savedUser = this.getUser();
    if (savedUser) {
      this.userSubject.next(savedUser);
    }
  }



googleBackendLogin(id_token: string) {
  return this.http.post(`${this.apiUrl}/google-login/`, { id_token }).pipe(
    tap((response: any) => this.saveAuthData(response)),
    catchError(err => throwError(() => err))
  );
}


  login(username: string, password: string) {
    return this.http.post(`${this.apiUrl}/login/`, { username, password }).pipe(
      tap((response: any) => {
        this.saveAuthData(response);
      }),
      catchError(err => throwError(() => err))
    );
  }

  private saveAuthData(data: any) {
    const cleanUser = {
      id: data.id,
      username: data.username,
      email: data.email,
      role: data.role,
      image:data.image,
    };

    localStorage.setItem("access", data.access);
    localStorage.setItem("refresh", data.refresh);
    localStorage.setItem("user", JSON.stringify(cleanUser));

    this.userSubject.next(cleanUser);
  }

  getUser() {
    return JSON.parse(localStorage.getItem("user") || "null");
  }

  getAccessToken() {
    return localStorage.getItem("access");
  }

  logout() {
    localStorage.removeItem("access");
    localStorage.removeItem("refresh");
    localStorage.removeItem("user");
    this.userSubject.next(null);
  }

  isLoggedIn() {
    return !!localStorage.getItem("access");
  }

  register(data: any) {
    return this.http.post(`${this.apiUrl}/register/`, data);
  }

  isRefreshing = false;
refreshQueue: ((token: string) => void)[] = [];

refreshTokenRequest() {
  const refresh = localStorage.getItem("refresh");
  return this.http.post(`${this.apiUrl}/token/refresh/`, { refresh });
}

enqueueRequest(callback: (token: string) => void) {
  this.refreshQueue.push(callback);
}

resolveQueuedRequests(token: string) {
  this.refreshQueue.forEach(cb => cb(token));
  this.refreshQueue = [];
}

setAccessToken(token: string) {
  localStorage.setItem("access", token);
}
updateUser(data: any) {
  // update BehaviorSubject
  this.userSubject.next(data);

  // update localStorage
  localStorage.setItem("user", JSON.stringify(data));
}


}
