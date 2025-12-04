import { HttpInterceptorFn, HttpRequest, HttpHandlerFn, HttpEvent } from "@angular/common/http";
import { inject } from "@angular/core";
import { Auth } from "../services/auth";
import { Observable, catchError, switchMap, throwError } from "rxjs";

export const AuthInterceptor: HttpInterceptorFn =
  (req: HttpRequest<any>, next: HttpHandlerFn): Observable<HttpEvent<any>> => {

    const auth = inject(Auth);
    const token = auth.getAccessToken();

    // PROTECTED ENDPOINTS (login required)
    const protectedEndpoints = [
      "/posts/",        // likes
      "/comment/",      // comments add/delete
      "/myposts/",      // create/edit/delete posts
      "/me",            // user profile
      "/update-profile"
    ];

    const isProtected = protectedEndpoints.some(url => req.url.includes(url));

    // BLOCK IF NOT LOGGED IN
    if (isProtected && !token) {
      alert("Login required to perform this action.");
      return throwError(() => ({
        status: 401,
        message: "Login required"
      }));
    }

    //  ATTACH TOKEN IF EXISTS
    let authReq = req;
    if (token) {
      authReq = req.clone({
        setHeaders: { Authorization: `Bearer ${token}` }
      });
    }

    // HANDLE TOKEN + AUTO REFRESH
    return next(authReq).pipe(
      catchError(err => {

        // 401 Unauthorized → try refresh
        if (err.status === 401 && localStorage.getItem("refresh")) {

          // BLOCK multiple refresh calls
          if (!auth.isRefreshing) {
            auth.isRefreshing = true;

            return auth.refreshTokenRequest().pipe(
              switchMap((res: any) => {
                const newToken = res.access;
                auth.setAccessToken(newToken);
                auth.isRefreshing = false;
                auth.resolveQueuedRequests(newToken);

                return next(
                  authReq.clone({
                    setHeaders: { Authorization: `Bearer ${newToken}` }
                  })
                );
              }),
              catchError(refreshErr => {
                auth.isRefreshing = false;
                auth.logout();
                return throwError(() => refreshErr);
              })
            );

          } else {
            // Queue requests during refresh
            return new Observable<HttpEvent<any>>(observer => {
              auth.enqueueRequest((newToken: string) => {
                next(
                  authReq.clone({
                    setHeaders: { Authorization: `Bearer ${newToken}` }
                  })
                ).subscribe({
                  next: v => observer.next(v),
                  error: e => observer.error(e),
                  complete: () => observer.complete()
                });
              });
            });
          }
        }

        // Other errors → throw normally
        return throwError(() => err);
      })
    );
  };
