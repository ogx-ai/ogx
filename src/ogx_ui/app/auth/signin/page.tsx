"use client";

import { signIn, signOut, useSession } from "next-auth/react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Copy, Check, Home } from "lucide-react";
import { useState } from "react";
import { useRouter } from "next/navigation";

// lucide-react 1.x removed brand icons, so the GitHub mark is inlined here.
function GithubIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.55 0-.27-.01-1.17-.02-2.12-3.2.7-3.87-1.36-3.87-1.36-.52-1.33-1.28-1.68-1.28-1.68-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.19 1.76 1.19 1.03 1.76 2.7 1.25 3.35.96.1-.75.4-1.25.72-1.54-2.55-.29-5.23-1.28-5.23-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.18 1.18a11.1 11.1 0 0 1 2.9-.39c.98 0 1.97.13 2.9.39 2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.41-2.69 5.38-5.25 5.67.41.36.78 1.06.78 2.14 0 1.55-.01 2.79-.01 3.17 0 .31.21.67.8.55A11.51 11.51 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z" />
    </svg>
  );
}

export default function SignInPage() {
  const { data: session, status } = useSession();
  const [copied, setCopied] = useState(false);
  const router = useRouter();

  const handleCopyToken = async () => {
    if (session?.accessToken) {
      await navigator.clipboard.writeText(session.accessToken);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (status === "loading") {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-muted-foreground">Loading...</div>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center min-h-screen">
      <Card className="w-[400px]">
        <CardHeader>
          <CardTitle>Authentication</CardTitle>
          <CardDescription>
            {session
              ? "You are successfully authenticated!"
              : "Sign in with GitHub to use your access token as an API key"}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!session ? (
            <Button
              onClick={() => {
                console.log("Signing in with GitHub...");
                signIn("github", { callbackUrl: "/auth/signin" }).catch(
                  error => {
                    console.error("Sign in error:", error);
                  }
                );
              }}
              className="w-full"
              variant="default"
            >
              <GithubIcon className="mr-2 h-4 w-4" />
              Sign in with GitHub
            </Button>
          ) : (
            <div className="space-y-4">
              <div className="text-sm text-muted-foreground">
                Signed in as {session.user?.email}
              </div>

              {session.accessToken && (
                <div className="space-y-2">
                  <div className="text-sm font-medium">
                    GitHub Access Token:
                  </div>
                  <div className="flex gap-2">
                    <code className="flex-1 p-2 bg-muted rounded text-xs break-all">
                      {session.accessToken}
                    </code>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={handleCopyToken}
                    >
                      {copied ? (
                        <Check className="h-4 w-4" />
                      ) : (
                        <Copy className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    This GitHub token will be used as your API key for
                    authenticated OGX requests.
                  </div>
                </div>
              )}

              <div className="flex gap-2">
                <Button onClick={() => router.push("/")} className="flex-1">
                  <Home className="mr-2 h-4 w-4" />
                  Go to Dashboard
                </Button>
                <Button
                  onClick={() => signOut()}
                  variant="outline"
                  className="flex-1"
                >
                  Sign out
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
